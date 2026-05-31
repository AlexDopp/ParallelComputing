/*
 * snow_simulation.cu
 *
 * CUDA C++ 2D Schnee-Simulation mit OpenGL-Rendering
 *
 * Kompilieren:
 *   nvcc snow_simulation.cu -o snow_sim \
 *       -lGL -lGLU -lglut -lGLEW \
 *       -arch=sm_75
 *
 * Steuerung:
 *   ESC / q  – Beenden
 *   r        – Restart
 */

#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <GL/freeglut.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <chrono>
#include <random>
#include <curand_kernel.h>

// ─────────────────────────────────────────────────────────────────────────────
//  Konfiguration
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int   THREADS_PER_BLOCK = 128;
static constexpr int   WIN_W             = 900;
static constexpr int   WIN_H             = 700;

static constexpr float DT           = 0.002f;
static constexpr float GRAVITY      = 0.12f;
static constexpr float WIND_BASE    = 0.120f;
static constexpr float FLAKE_RADIUS = 0.004f;

static const int FLAKE_COUNTS[]  = { 512, 1024, 2048, 4096, 8192, 16384 };
static const int NUM_FLAKE_STEPS = 6;
static       int g_flakeStepIdx  = 3;
static       int g_numFlakes     = FLAKE_COUNTS[3];

static bool g_uncapFps = false;
static bool g_useCPU   = false;   // false = GPU (Standard)

#ifdef _WIN32
typedef BOOL (WINAPI* PFNWGLSWAPINTERVALEXTPROC)(int);
static PFNWGLSWAPINTERVALEXTPROC g_wglSwapInterval = nullptr;
#endif

static void applyVsync()
{
#ifdef _WIN32
    if(g_wglSwapInterval) g_wglSwapInterval(g_uncapFps ? 0 : 1);
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
//  Szenen-Geometrie
// ─────────────────────────────────────────────────────────────────────────────
static constexpr float GROUND_Y      = 0.12f;
static constexpr float HOUSE_X0      = 0.42f;
static constexpr float HOUSE_X1      = 0.74f;
static constexpr float HOUSE_WALL_Y0 = GROUND_Y;
static constexpr float HOUSE_WALL_Y1 = 0.48f;
static constexpr float ROOF_PEAK_X   = (HOUSE_X0 + HOUSE_X1) * 0.5f;
static constexpr float ROOF_PEAK_Y   = 0.70f;
static constexpr float TREE_X        = 0.18f;
static constexpr float TREE_TRUNK_X0 = TREE_X - 0.018f;
static constexpr float TREE_TRUNK_X1 = TREE_X + 0.018f;
static constexpr float TREE_TRUNK_Y0 = GROUND_Y;
static constexpr float TREE_TRUNK_Y1 = GROUND_Y + 0.10f;

struct TriLayer { float x0, x1, yBase, yTip; };

static constexpr TriLayer TREE_LAYERS_HOST[3] = {
    { TREE_X-0.10f,  TREE_X+0.10f,  GROUND_Y+0.06f, GROUND_Y+0.22f },
    { TREE_X-0.075f, TREE_X+0.075f, GROUND_Y+0.17f, GROUND_Y+0.33f },
    { TREE_X-0.05f,  TREE_X+0.05f,  GROUND_Y+0.28f, GROUND_Y+0.44f },
};
__device__ __constant__ TriLayer TREE_LAYERS[3] = {
    { TREE_X-0.10f,  TREE_X+0.10f,  GROUND_Y+0.06f, GROUND_Y+0.22f },
    { TREE_X-0.075f, TREE_X+0.075f, GROUND_Y+0.17f, GROUND_Y+0.33f },
    { TREE_X-0.05f,  TREE_X+0.05f,  GROUND_Y+0.28f, GROUND_Y+0.44f },
};

// ─────────────────────────────────────────────────────────────────────────────
//  Flake
// ─────────────────────────────────────────────────────────────────────────────
struct Flake { float x,y,vx,vy,radius,spawnDelay,fallSpeed; };

// ─────────────────────────────────────────────────────────────────────────────
//  GPU: Device-Hilfsfunktionen
// ─────────────────────────────────────────────────────────────────────────────
__device__ float gpu_pointSegDist(float px,float py,float ax,float ay,float bx,float by,float&nx,float&ny)
{
    float dx=bx-ax,dy=by-ay,len2=dx*dx+dy*dy,t=0.f;
    if(len2>1e-9f) t=fminf(1.f,fmaxf(0.f,((px-ax)*dx+(py-ay)*dy)/len2));
    float cx=ax+t*dx,cy=ay+t*dy,ex=px-cx,ey=py-cy,d=sqrtf(ex*ex+ey*ey);
    if(d>1e-9f){nx=ex/d;ny=ey/d;}else{nx=0.f;ny=1.f;}
    return d;
}
__device__ bool gpu_collideRoof(float x,float y,float r,float lx,float ly,float rx2,float ry,float px,float py,float&nx,float&ny,float&pen)
{
    float n1x,n1y,d1=gpu_pointSegDist(x,y,lx,ly,px,py,n1x,n1y);
    float n2x,n2y,d2=gpu_pointSegDist(x,y,rx2,ry,px,py,n2x,n2y);
    float n3x,n3y,d3=gpu_pointSegDist(x,y,lx,ly,rx2,ry,n3x,n3y);
    float d=fminf(fminf(d1,d2),d3);
    if(d<r){pen=r-d;
        if(d<=d2&&d<=d3){nx=n1x;ny=n1y;}
        else if(d2<=d1&&d2<=d3){nx=n2x;ny=n2y;}
        else{nx=n3x;ny=n3y;}
        return true;}
    return false;
}
__device__ bool gpu_collideAABB(float x,float y,float r,float bx0,float by0,float bx1,float by1,float&nx,float&ny,float&pen)
{
    float cx=fmaxf(bx0,fminf(x,bx1)),cy=fmaxf(by0,fminf(y,by1));
    float dx=x-cx,dy=y-cy,dist=sqrtf(dx*dx+dy*dy);
    if(dist<1e-9f){
        float dl=x-bx0,dr=bx1-x,db=y-by0,dt=by1-y,mn=fminf(fminf(dl,dr),fminf(db,dt));
        if(mn==dl){nx=-1.f;ny=0.f;pen=r+dl;}
        else if(mn==dr){nx=1.f;ny=0.f;pen=r+dr;}
        else if(mn==db){nx=0.f;ny=-1.f;pen=r+db;}
        else{nx=0.f;ny=1.f;pen=r+dt;}
        return true;}
    if(dist<r){pen=r-dist;nx=dx/dist;ny=dy/dist;return true;}
    return false;
}
__device__ bool gpu_pointInTri(float px,float py,float ax,float ay,float bx,float by,float cx,float cy)
{
    float d1=(px-bx)*(ay-by)-(ax-bx)*(py-by);
    float d2=(px-cx)*(by-cy)-(bx-cx)*(py-cy);
    float d3=(px-ax)*(cy-ay)-(cx-ax)*(py-ay);
    return !((d1<0||d2<0||d3<0)&&(d1>0||d2>0||d3>0));
}
__device__ bool gpu_insideHouseRoof(float x,float y)
{
    float m=0.003f;
    return gpu_pointInTri(x,y,HOUSE_X0+m,HOUSE_WALL_Y1,HOUSE_X1-m,HOUSE_WALL_Y1,ROOF_PEAK_X,ROOF_PEAK_Y-m);
}

// ─────────────────────────────────────────────────────────────────────────────
//  CPU: Host-Hilfsfunktionen  (identische Logik, kein __device__)
// ─────────────────────────────────────────────────────────────────────────────
static inline float cpu_pointSegDist(float px,float py,float ax,float ay,float bx,float by,float&nx,float&ny)
{
    float dx=bx-ax,dy=by-ay,len2=dx*dx+dy*dy,t=0.f;
    if(len2>1e-9f) t=fminf(1.f,fmaxf(0.f,((px-ax)*dx+(py-ay)*dy)/len2));
    float cx=ax+t*dx,cy=ay+t*dy,ex=px-cx,ey=py-cy,d=sqrtf(ex*ex+ey*ey);
    if(d>1e-9f){nx=ex/d;ny=ey/d;}else{nx=0.f;ny=1.f;}
    return d;
}
static inline bool cpu_collideRoof(float x,float y,float r,float lx,float ly,float rx2,float ry,float px,float py,float&nx,float&ny,float&pen)
{
    float n1x,n1y,d1=cpu_pointSegDist(x,y,lx,ly,px,py,n1x,n1y);
    float n2x,n2y,d2=cpu_pointSegDist(x,y,rx2,ry,px,py,n2x,n2y);
    float n3x,n3y,d3=cpu_pointSegDist(x,y,lx,ly,rx2,ry,n3x,n3y);
    float d=fminf(fminf(d1,d2),d3);
    if(d<r){pen=r-d;
        if(d<=d2&&d<=d3){nx=n1x;ny=n1y;}
        else if(d2<=d1&&d2<=d3){nx=n2x;ny=n2y;}
        else{nx=n3x;ny=n3y;}
        return true;}
    return false;
}
static inline bool cpu_collideAABB(float x,float y,float r,float bx0,float by0,float bx1,float by1,float&nx,float&ny,float&pen)
{
    float cx=fmaxf(bx0,fminf(x,bx1)),cy=fmaxf(by0,fminf(y,by1));
    float dx=x-cx,dy=y-cy,dist=sqrtf(dx*dx+dy*dy);
    if(dist<1e-9f){
        float dl=x-bx0,dr=bx1-x,db=y-by0,dt=by1-y,mn=fminf(fminf(dl,dr),fminf(db,dt));
        if(mn==dl){nx=-1.f;ny=0.f;pen=r+dl;}
        else if(mn==dr){nx=1.f;ny=0.f;pen=r+dr;}
        else if(mn==db){nx=0.f;ny=-1.f;pen=r+db;}
        else{nx=0.f;ny=1.f;pen=r+dt;}
        return true;}
    if(dist<r){pen=r-dist;nx=dx/dist;ny=dy/dist;return true;}
    return false;
}
static inline bool cpu_pointInTri(float px,float py,float ax,float ay,float bx,float by,float cx,float cy)
{
    float d1=(px-bx)*(ay-by)-(ax-bx)*(py-by);
    float d2=(px-cx)*(by-cy)-(bx-cx)*(py-cy);
    float d3=(px-ax)*(cy-ay)-(cx-ax)*(py-ay);
    return !((d1<0||d2<0||d3<0)&&(d1>0||d2>0||d3>0));
}
static inline bool cpu_insideHouseRoof(float x,float y)
{
    float m=0.003f;
    return cpu_pointInTri(x,y,HOUSE_X0+m,HOUSE_WALL_Y1,HOUSE_X1-m,HOUSE_WALL_Y1,ROOF_PEAK_X,ROOF_PEAK_Y-m);
}

// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Kernel
// ─────────────────────────────────────────────────────────────────────────────
__global__ void initRngKernel(curandState*rng,int n,unsigned long long seed)
{
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=n)return;
    curand_init(seed+id*6364136223846793005ULL,id,0,&rng[id]);
}
__global__ void initFlakesKernel(Flake*flakes,curandState*rng,int n)
{
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=n)return;
    curandState&r=rng[id];
    flakes[id]={curand_uniform(&r),curand_uniform(&r),
                (curand_uniform(&r)-0.5f)*0.005f,
                -(curand_uniform(&r)*0.002f+0.0015f),
                FLAKE_RADIUS*(0.6f+curand_uniform(&r)*0.8f),
                0.f,1.f+(int)(curand_uniform(&r)*10.9999f)*0.05f};
}
__global__ void updateFlakesKernel(Flake*flakes,curandState*rng,int n,float dt,float wind)
{
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=n)return;
    Flake f=flakes[id];
    curandState&r=rng[id];
    if(f.x<-5.f)return;
    if(f.spawnDelay>0.f){f.spawnDelay-=dt;flakes[id]=f;return;}

    float pw=wind*0.1f+(curand_uniform(&r)-0.5f)*wind*4.f;
    float tx=(curand_uniform(&r)-0.5f)*0.003f,ty=(curand_uniform(&r)-0.5f)*0.001f;
    f.vx+=(pw+tx)*dt; f.vy-=GRAVITY*f.fallSpeed*dt+ty*dt;
    f.vx*=0.98f; f.vy*=0.99f;
    float mvy=-0.003f*f.fallSpeed;
    if(f.vy<mvy)f.vy=mvy;
    if(f.vx>0.004f)f.vx=0.004f; if(f.vx<-0.004f)f.vx=-0.004f;
    f.x+=f.vx; f.y+=f.vy;
    if(f.x<0.f)f.x+=1.f; if(f.x>1.f)f.x-=1.f;

    auto respawn=[&](){
        f.x=curand_uniform(&r);f.y=1.02f;
        f.vx=(curand_uniform(&r)-0.5f)*0.005f;
        f.vy=-(curand_uniform(&r)*0.002f+0.0015f);
        f.spawnDelay=curand_uniform(&r)*1.f;
        f.fallSpeed=1.f+(int)(curand_uniform(&r)*10.9999f)*0.05f;
    };
    if(f.y<-0.02f){
        if(curand_uniform(&r)<0.01f){f.x=-10.f;f.y=-10.f;f.vx=f.vy=0.f;f.spawnDelay=0.f;flakes[id]=f;return;}
        respawn();flakes[id]=f;return;
    }
    float nx,ny,pen,res=0.05f;
    auto resolve=[&](float cnx,float cny,float cpen){
        f.x+=cnx*cpen;f.y+=cny*cpen;
        float vn=f.vx*cnx+f.vy*cny;
        if(vn<0.f){f.vx-=(1.f+res)*vn*cnx;f.vy-=(1.f+res)*vn*cny;}
    };
    for(int it=0;it<3;it++){
        if(gpu_collideAABB(f.x,f.y,f.radius,HOUSE_X0,HOUSE_WALL_Y0,HOUSE_X1,HOUSE_WALL_Y1,nx,ny,pen))resolve(nx,ny,pen);
        if(gpu_collideRoof(f.x,f.y,f.radius,HOUSE_X0,HOUSE_WALL_Y1,HOUSE_X1,HOUSE_WALL_Y1,ROOF_PEAK_X,ROOF_PEAK_Y,nx,ny,pen))resolve(nx,ny,pen);
        if(gpu_collideAABB(f.x,f.y,f.radius,TREE_TRUNK_X0,TREE_TRUNK_Y0,TREE_TRUNK_X1,TREE_TRUNK_Y1,nx,ny,pen))resolve(nx,ny,pen);
        for(int i=0;i<3;i++){const TriLayer&L=TREE_LAYERS[i];
            if(gpu_collideRoof(f.x,f.y,f.radius,L.x0,L.yBase,L.x1,L.yBase,TREE_X,L.yTip,nx,ny,pen))resolve(nx,ny,pen);}
    }
    if(gpu_insideHouseRoof(f.x,f.y)){respawn();flakes[id]=f;return;}
    for(int i=0;i<3;i++){const TriLayer&L=TREE_LAYERS[i];float m=0.004f;
        if(gpu_pointInTri(f.x,f.y,L.x0+m,L.yBase,L.x1-m,L.yBase,TREE_X,L.yTip-m)){respawn();flakes[id]=f;return;}}
    flakes[id]=f;
}

// ─────────────────────────────────────────────────────────────────────────────
//  CPU-Update  (single-threaded, identische Physik)
// ─────────────────────────────────────────────────────────────────────────────
// Forward-Deklarationen damit cpuInitFlakes/cpuUpdateFlakes h_flakes kennen
static Flake*    h_flakes    = nullptr;

// Pro Flocke ein LCG-State (64-bit, selbe Konstante wie CUDA)
static uint64_t* h_rngStates = nullptr;

static inline float lcgUniform(uint64_t& s)
{
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((s >> 33) & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

static void cpuInitFlakes()
{
    uint64_t seed = (uint64_t)time(nullptr);
    for(int id=0;id<g_numFlakes;id++){
        h_rngStates[id] = seed + (uint64_t)id * 6364136223846793005ULL;
        uint64_t& s = h_rngStates[id];
        int step=(int)(lcgUniform(s)*10.9999f);
        h_flakes[id]={lcgUniform(s),lcgUniform(s),
                      (lcgUniform(s)-0.5f)*0.005f,
                      -(lcgUniform(s)*0.002f+0.0015f),
                      FLAKE_RADIUS*(0.6f+lcgUniform(s)*0.8f),
                      0.f, 1.f+step*0.05f};
    }
}

static void cpuUpdateFlakes(float dt, float wind)
{
    for(int id=0;id<g_numFlakes;id++){
        Flake& f=h_flakes[id];
        uint64_t& s=h_rngStates[id];
        if(f.x<-5.f)continue;
        if(f.spawnDelay>0.f){f.spawnDelay-=dt;continue;}

        float pw=wind*0.1f+(lcgUniform(s)-0.5f)*wind*4.f;
        float tx=(lcgUniform(s)-0.5f)*0.003f,ty=(lcgUniform(s)-0.5f)*0.001f;
        f.vx+=(pw+tx)*dt; f.vy-=GRAVITY*f.fallSpeed*dt+ty*dt;
        f.vx*=0.98f; f.vy*=0.99f;
        float mvy=-0.003f*f.fallSpeed;
        if(f.vy<mvy)f.vy=mvy;
        if(f.vx>0.004f)f.vx=0.004f; if(f.vx<-0.004f)f.vx=-0.004f;
        f.x+=f.vx; f.y+=f.vy;
        if(f.x<0.f)f.x+=1.f; if(f.x>1.f)f.x-=1.f;

        auto respawn=[&](){
            f.x=lcgUniform(s);f.y=1.02f;
            f.vx=(lcgUniform(s)-0.5f)*0.005f;
            f.vy=-(lcgUniform(s)*0.002f+0.0015f);
            f.spawnDelay=lcgUniform(s)*1.f;
            f.fallSpeed=1.f+(int)(lcgUniform(s)*10.9999f)*0.05f;
        };
        if(f.y<-0.02f){
            if(lcgUniform(s)<0.01f){f.x=-10.f;f.y=-10.f;f.vx=f.vy=0.f;f.spawnDelay=0.f;continue;}
            respawn();continue;
        }
        float nx,ny,pen,res=0.05f;
        auto resolve=[&](float cnx,float cny,float cpen){
            f.x+=cnx*cpen;f.y+=cny*cpen;
            float vn=f.vx*cnx+f.vy*cny;
            if(vn<0.f){f.vx-=(1.f+res)*vn*cnx;f.vy-=(1.f+res)*vn*cny;}
        };
        for(int it=0;it<3;it++){
            if(cpu_collideAABB(f.x,f.y,f.radius,HOUSE_X0,HOUSE_WALL_Y0,HOUSE_X1,HOUSE_WALL_Y1,nx,ny,pen))resolve(nx,ny,pen);
            if(cpu_collideRoof(f.x,f.y,f.radius,HOUSE_X0,HOUSE_WALL_Y1,HOUSE_X1,HOUSE_WALL_Y1,ROOF_PEAK_X,ROOF_PEAK_Y,nx,ny,pen))resolve(nx,ny,pen);
            if(cpu_collideAABB(f.x,f.y,f.radius,TREE_TRUNK_X0,TREE_TRUNK_Y0,TREE_TRUNK_X1,TREE_TRUNK_Y1,nx,ny,pen))resolve(nx,ny,pen);
            for(int i=0;i<3;i++){const TriLayer&L=TREE_LAYERS_HOST[i];
                if(cpu_collideRoof(f.x,f.y,f.radius,L.x0,L.yBase,L.x1,L.yBase,TREE_X,L.yTip,nx,ny,pen))resolve(nx,ny,pen);}
        }
        if(cpu_insideHouseRoof(f.x,f.y)){respawn();continue;}
        for(int i=0;i<3;i++){const TriLayer&L=TREE_LAYERS_HOST[i];float m=0.004f;
            if(cpu_pointInTri(f.x,f.y,L.x0+m,L.yBase,L.x1-m,L.yBase,TREE_X,L.yTip-m)){respawn();continue;}}
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Host-Zustand
// ─────────────────────────────────────────────────────────────────────────────
static Flake*       d_flakes = nullptr;
// h_flakes bereits oben vor CPU-Funktionen deklariert
static curandState* d_rng    = nullptr;
static int          g_frame  = 0;

// Performance
static float g_fps=0.f,g_kernelMs=0.f,g_frameMs=0.f;
static float g_kernelMsAcc=0.f,g_frameMsAcc=0.f;
static int   g_perfFrameAcc=0,g_perfTimeAcc=0;

// CUDA-Events
static cudaEvent_t g_evFrameStart=nullptr,g_evFrameStop=nullptr;
static cudaEvent_t g_evKernelStart=nullptr,g_evKernelStop=nullptr;

// ─────────────────────────────────────────────────────────────────────────────
//  Allokation / Init  (GPU oder CPU je nach Flag)
// ─────────────────────────────────────────────────────────────────────────────
static void allocAndInit()
{
    // Aufräumen
    if(d_flakes){cudaFree(d_flakes);d_flakes=nullptr;}
    if(d_rng)   {cudaFree(d_rng);   d_rng=nullptr;}
    if(h_flakes){delete[]h_flakes;  h_flakes=nullptr;}
    if(h_rngStates){delete[]h_rngStates;h_rngStates=nullptr;}

    h_flakes = new Flake[g_numFlakes];

    if(g_useCPU){
        // CPU-Pfad: nur Host-Speicher
        h_rngStates = new uint64_t[g_numFlakes];
        cpuInitFlakes();
    } else {
        // GPU-Pfad
        cudaMalloc(&d_flakes,g_numFlakes*sizeof(Flake));
        cudaMalloc(&d_rng,   g_numFlakes*sizeof(curandState));
        int blocks=(g_numFlakes+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
        initRngKernel<<<blocks,THREADS_PER_BLOCK>>>(d_rng,g_numFlakes,(unsigned long long)time(nullptr));
        cudaDeviceSynchronize();
        initFlakesKernel<<<blocks,THREADS_PER_BLOCK>>>(d_flakes,d_rng,g_numFlakes);
        cudaDeviceSynchronize();

        if(!g_evFrameStart){
            cudaEventCreate(&g_evFrameStart); cudaEventCreate(&g_evFrameStop);
            cudaEventCreate(&g_evKernelStart);cudaEventCreate(&g_evKernelStop);
        }
    }

    g_frame=0; g_fps=0.f; g_kernelMs=0.f; g_frameMs=0.f;
    g_kernelMsAcc=0.f; g_frameMsAcc=0.f; g_perfFrameAcc=0; g_perfTimeAcc=0;
}

// ─────────────────────────────────────────────────────────────────────────────
//  OpenGL-Helfer
// ─────────────────────────────────────────────────────────────────────────────
static void glowLine(float x0,float y0,float x1,float y1,float r,float g,float b)
{
    glLineWidth(6.f);glColor4f(r,g,b,0.08f);glBegin(GL_LINES);glVertex2f(x0,y0);glVertex2f(x1,y1);glEnd();
    glLineWidth(3.5f);glColor4f(r,g,b,0.20f);glBegin(GL_LINES);glVertex2f(x0,y0);glVertex2f(x1,y1);glEnd();
    glLineWidth(1.8f);glColor4f(r,g,b,0.55f);glBegin(GL_LINES);glVertex2f(x0,y0);glVertex2f(x1,y1);glEnd();
    glLineWidth(0.8f);glColor4f(1.f,1.f,1.f,0.90f);glBegin(GL_LINES);glVertex2f(x0,y0);glVertex2f(x1,y1);glEnd();
}
static void glowRect(float x0,float y0,float x1,float y1,float r,float g,float b)
{glowLine(x0,y0,x1,y0,r,g,b);glowLine(x1,y0,x1,y1,r,g,b);glowLine(x1,y1,x0,y1,r,g,b);glowLine(x0,y1,x0,y0,r,g,b);}
static void glowTri(float ax,float ay,float bx,float by,float cx,float cy,float r,float g,float b)
{glowLine(ax,ay,bx,by,r,g,b);glowLine(bx,by,cx,cy,r,g,b);glowLine(cx,cy,ax,ay,r,g,b);}

static void setPixelProj()
{glMatrixMode(GL_PROJECTION);glLoadIdentity();glOrtho(0,WIN_W,WIN_H,0,-1,1);glMatrixMode(GL_MODELVIEW);glLoadIdentity();}
static void setSceneProj()
{glMatrixMode(GL_PROJECTION);glLoadIdentity();glOrtho(0,1,0,1,-1,1);glMatrixMode(GL_MODELVIEW);glLoadIdentity();}
static void drawText(float px,float py,const char*s,void*font=GLUT_BITMAP_8_BY_13)
{glRasterPos2f(px,py);for(const char*c=s;*c;c++)glutBitmapCharacter(font,*c);}
static void fillPx(float x0,float y0,float x1,float y1)
{glBegin(GL_QUADS);glVertex2f(x0,y0);glVertex2f(x1,y0);glVertex2f(x1,y1);glVertex2f(x0,y1);glEnd();}
static void strokePx(float x0,float y0,float x1,float y1)
{glBegin(GL_LINE_LOOP);glVertex2f(x0,y0);glVertex2f(x1,y0);glVertex2f(x1,y1);glVertex2f(x0,y1);glEnd();}

// Einfacher Toggle-Switch (für FPS Uncap)
static void drawSwitch(float cx,float cy,bool on)
{
    float hw=18.f,hh=8.f;
    float x0=cx-hw,y0=cy-hh,x1=cx+hw,y1=cy+hh;
    glColor4f(on?0.2f:0.25f,on?0.7f:0.25f,on?0.2f:0.35f,0.85f);
    fillPx(x0,y0,x1,y1);
    glColor4f(0.5f,0.5f,0.8f,0.7f); strokePx(x0,y0,x1,y1);
    float kx=on?cx+hw-hh:cx-hw+hh,kr=hh-1.f;
    glColor4f(1.f,1.f,1.f,0.95f);
    fillPx(kx-kr,cy-kr,kx+kr,cy+kr);
}

// CPU/GPU-Switch: nur Track + Knob, Labels werden außen in drawMenu gezeichnet
// g_cpuGpuLockUntil: glutGet-Zeitstempel bis zu dem der Switch gesperrt ist
static int g_cpuGpuLockUntil = 0;

static void drawCpuGpuSwitch(float cx, float cy)
{
    float hw=18.f, hh=9.f;
    float x0=cx-hw,y0=cy-hh,x1=cx+hw,y1=cy+hh;

    bool locked = (glutGet(GLUT_ELAPSED_TIME) < g_cpuGpuLockUntil);
    float dim = locked ? 0.5f : 1.0f;
    if(g_useCPU) glColor4f(0.7f*dim,0.35f*dim,0.05f*dim,0.85f);
    else         glColor4f(0.1f*dim,0.35f*dim,0.75f*dim,0.85f);
    fillPx(x0,y0,x1,y1);
    glColor4f(0.6f,0.6f,0.9f,0.7f); strokePx(x0,y0,x1,y1);

    float kx = g_useCPU ? cx+hw-hh : cx-hw+hh;
    float kr  = hh-1.5f;
    glColor4f(dim,dim,dim,0.95f);
    fillPx(kx-kr,cy-kr,kx+kr,cy+kr);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Menü-Layout
// ─────────────────────────────────────────────────────────────────────────────
static constexpr float MX=8.f,MY=8.f,MW=230.f;

static constexpr float ROW_TITLE  = MY+18.f;
static constexpr float ROW_FPS    = MY+38.f;
static constexpr float ROW_KERNEL = MY+55.f;
static constexpr float ROW_FRAME  = MY+72.f;
static constexpr float ROW_DIV1   = MY+84.f;
static constexpr float ROW_UNCAP  = MY+100.f;
static constexpr float ROW_CPUGPU = MY+120.f;
static constexpr float ROW_DIV2   = MY+134.f;
static constexpr float ROW_FLAKES = MY+149.f;
static constexpr float ROW_STEPS  = MY+168.f;
static constexpr float ROW_BTN    = MY+194.f;
static constexpr float MH         = ROW_BTN+20.f-MY;

static constexpr float BTN_X0=MX+8.f,BTN_X1=MX+MW-8.f;
static constexpr float BTN_Y0=ROW_BTN-12.f,BTN_Y1=ROW_BTN+12.f;
static constexpr float STEP_Y0=ROW_STEPS-10.f,STEP_Y1=ROW_STEPS+10.f;
static constexpr float STEP_W=(MW-16.f)/NUM_FLAKE_STEPS;

// Switch-Mittelpunkte
static constexpr float SW_UNCAP_CX  = MX+MW-26.f,  SW_UNCAP_CY  = ROW_UNCAP;
// CPU/GPU-Switch: mittig im Panel, Platz für Labels links ("GPU") und rechts ("CPU")
static constexpr float SW_CPUGPU_CX = MX+MW/2.f,   SW_CPUGPU_CY = ROW_CPUGPU;

static bool inRect(float mx,float my,float x0,float y0,float x1,float y1)
{return mx>=x0&&mx<=x1&&my>=y0&&my<=y1;}
static bool inStep(float mx,float my,int i)
{float x0=MX+8.f+i*STEP_W,x1=x0+STEP_W-2.f;return inRect(mx,my,x0,STEP_Y0,x1,STEP_Y1);}
static bool inSwitch(float mx,float my,float cx,float cy,float hw,float hh)
{return inRect(mx,my,cx-hw,cy-hh,cx+hw,cy+hh);}

// ─────────────────────────────────────────────────────────────────────────────
//  Menü zeichnen
// ─────────────────────────────────────────────────────────────────────────────
static void drawMenu()
{
    setPixelProj();
    glDisable(GL_LINE_SMOOTH);
    glLineWidth(1.f);

    glColor4f(0.05f,0.05f,0.12f,0.85f); fillPx(MX,MY,MX+MW,MY+MH);
    glColor4f(0.3f,0.5f,1.f,0.6f);      strokePx(MX,MY,MX+MW,MY+MH);

    char buf[80];

    // Titel
    glColor4f(0.6f,0.8f,1.f,1.f);
    drawText(MX+8.f,ROW_TITLE+4.f,"CUDA Snow Sim",GLUT_BITMAP_8_BY_13);

    // FPS
    snprintf(buf,sizeof(buf),"FPS:    %.1f",g_fps);
    glColor4f(0.4f,1.f,0.5f,1.f);
    drawText(MX+8.f,ROW_FPS+4.f,buf,GLUT_BITMAP_8_BY_13);

    // Kernel
    snprintf(buf,sizeof(buf),"Kernel: %.3f ms",g_kernelMs);
    glColor4f(1.f,0.75f,0.2f,1.f);
    drawText(MX+8.f,ROW_KERNEL+4.f,buf,GLUT_BITMAP_8_BY_13);

    // Frame
    snprintf(buf,sizeof(buf),"Frame:  %.3f ms",g_frameMs);
    glColor4f(1.f,0.45f,0.45f,1.f);
    drawText(MX+8.f,ROW_FRAME+4.f,buf,GLUT_BITMAP_8_BY_13);

    // Trennlinie 1
    glColor4f(0.3f,0.4f,0.7f,0.5f);
    glBegin(GL_LINES);glVertex2f(MX+6.f,ROW_DIV1);glVertex2f(MX+MW-6.f,ROW_DIV1);glEnd();

    // FPS Uncapped
    glColor4f(0.85f,0.85f,0.85f,1.f);
    drawText(MX+8.f,ROW_UNCAP+4.f,"FPS Uncapped",GLUT_BITMAP_8_BY_13);
    drawSwitch(SW_UNCAP_CX,SW_UNCAP_CY,g_uncapFps);

    // CPU/GPU Switch – Label "GPU" links, Switch mittig, "CPU" rechts
    {
        bool locked=(glutGet(GLUT_ELAPSED_TIME)<g_cpuGpuLockUntil);
        // "GPU" links vom Switch
        glColor4f(g_useCPU?0.45f:1.f, g_useCPU?0.45f:1.f, g_useCPU?0.45f:1.f, 1.f);
        drawText(SW_CPUGPU_CX-18.f-28.f, ROW_CPUGPU+4.f, "GPU", GLUT_BITMAP_8_BY_13);
        // Switch selbst
        drawCpuGpuSwitch(SW_CPUGPU_CX,SW_CPUGPU_CY);
        // "CPU" rechts vom Switch
        glColor4f(g_useCPU?1.f:0.45f, g_useCPU?0.85f:0.45f, g_useCPU?0.45f:0.45f, 1.f);
        drawText(SW_CPUGPU_CX+18.f+4.f,  ROW_CPUGPU+4.f, "CPU", GLUT_BITMAP_8_BY_13);
        // Lock-Hinweis (kleines "..." wenn gesperrt)
        if(locked){ glColor4f(0.6f,0.6f,0.6f,0.8f); drawText(SW_CPUGPU_CX-6.f,ROW_CPUGPU+4.f,"...",GLUT_BITMAP_8_BY_13); }
    }

    // Trennlinie 2
    glColor4f(0.3f,0.4f,0.7f,0.5f);
    glBegin(GL_LINES);glVertex2f(MX+6.f,ROW_DIV2);glVertex2f(MX+MW-6.f,ROW_DIV2);glEnd();

    // Flocken
    snprintf(buf,sizeof(buf),"Flakes: %d",g_numFlakes);
    glColor4f(0.9f,0.9f,0.9f,1.f);
    drawText(MX+8.f,ROW_FLAKES+4.f,buf,GLUT_BITMAP_8_BY_13);

    // Stufen-Leiste
    static const char*labels[]={"512","1k","2k","4k","8k","16k"};
    for(int i=0;i<NUM_FLAKE_STEPS;i++){
        float x0=MX+8.f+i*STEP_W,x1=x0+STEP_W-2.f;
        bool sel=(i==g_flakeStepIdx);
        glColor4f(sel?0.3f:0.12f,sel?0.6f:0.18f,sel?1.f:0.4f,sel?0.9f:0.8f);
        fillPx(x0,STEP_Y0,x1,STEP_Y1);
        glColor4f(sel?0.6f:0.4f,sel?0.85f:0.6f,1.f,0.9f);
        strokePx(x0,STEP_Y0,x1,STEP_Y1);
        glColor4f(sel?1.f:0.7f,sel?1.f:0.8f,1.f,1.f);
        drawText(x0+2.f,STEP_Y0+13.f,labels[i],GLUT_BITMAP_8_BY_13);
    }

    // Restart
    glColor4f(0.15f,0.45f,0.15f,0.88f); fillPx(BTN_X0,BTN_Y0,BTN_X1,BTN_Y1);
    glColor4f(0.35f,1.f,0.35f,1.f);     strokePx(BTN_X0,BTN_Y0,BTN_X1,BTN_Y1);
    glColor4f(1.f,1.f,1.f,1.f);
    drawText(BTN_X0+(BTN_X1-BTN_X0)/2.f-32.f,BTN_Y0+14.f,"[ RESTART ]",GLUT_BITMAP_8_BY_13);

    glEnable(GL_LINE_SMOOTH);
    setSceneProj();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Szene zeichnen
// ─────────────────────────────────────────────────────────────────────────────
static void drawScene()
{
    glColor4f(0.f,0.f,0.f,1.f);
    glBegin(GL_QUADS);glVertex2f(0,0);glVertex2f(1,0);glVertex2f(1,1);glVertex2f(0,1);glEnd();
    glowRect(HOUSE_X0,HOUSE_WALL_Y0,HOUSE_X1,HOUSE_WALL_Y1,1.f,0.55f,0.05f);
    glowRect(HOUSE_X0+0.04f,HOUSE_WALL_Y0+0.10f,HOUSE_X0+0.10f,HOUSE_WALL_Y0+0.18f,1.f,0.9f,0.2f);
    glowRect(HOUSE_X1-0.10f,HOUSE_WALL_Y0+0.10f,HOUSE_X1-0.04f,HOUSE_WALL_Y0+0.18f,1.f,0.9f,0.2f);
    glowRect(ROOF_PEAK_X-0.035f,HOUSE_WALL_Y0,ROOF_PEAK_X+0.035f,HOUSE_WALL_Y0+0.12f,0.85f,0.1f,0.7f);
    glowTri(HOUSE_X0,HOUSE_WALL_Y1,HOUSE_X1,HOUSE_WALL_Y1,ROOF_PEAK_X,ROOF_PEAK_Y,1.f,0.15f,0.15f);
    glowRect(TREE_TRUNK_X0,TREE_TRUNK_Y0,TREE_TRUNK_X1,TREE_TRUNK_Y1,0.8f,0.4f,0.05f);
    static const float tc[3][3]={{0.10f,1.f,0.35f},{0.15f,0.85f,0.50f},{0.20f,0.70f,0.65f}};
    for(int i=0;i<3;i++){const TriLayer&L=TREE_LAYERS_HOST[i];
        glowTri(L.x0,L.yBase,L.x1,L.yBase,TREE_X,L.yTip,tc[i][0],tc[i][1],tc[i][2]);}

    glEnable(GL_POINT_SMOOTH);
    glPointSize(7.f);glBegin(GL_POINTS);
    for(int i=0;i<g_numFlakes;i++){const Flake&f=h_flakes[i];
        if(f.x<-5.f||f.spawnDelay>0.f)continue;glColor4f(0.7f,0.92f,1.f,0.12f);glVertex2f(f.x,f.y);}glEnd();
    glPointSize(4.f);glBegin(GL_POINTS);
    for(int i=0;i<g_numFlakes;i++){const Flake&f=h_flakes[i];
        if(f.x<-5.f||f.spawnDelay>0.f)continue;glColor4f(0.8f,0.95f,1.f,0.35f);glVertex2f(f.x,f.y);}glEnd();
    glPointSize(2.f);glBegin(GL_POINTS);
    for(int i=0;i<g_numFlakes;i++){const Flake&f=h_flakes[i];
        if(f.x<-5.f||f.spawnDelay>0.f)continue;glColor4f(1.f,1.f,1.f,0.95f);glVertex2f(f.x,f.y);}glEnd();
}

// ─────────────────────────────────────────────────────────────────────────────
//  GLUT-Callbacks
// ─────────────────────────────────────────────────────────────────────────────
static void doRestart()
{
    g_numFlakes=FLAKE_COUNTS[g_flakeStepIdx];
    allocAndInit();
}

static void display()
{
    float kMs=0.f,fMs=0.f;

    if(g_useCPU){
        // CPU-Pfad: chrono für Zeitmessung
        auto t0=std::chrono::high_resolution_clock::now();
        cpuUpdateFlakes(DT,WIND_BASE);
        auto t1=std::chrono::high_resolution_clock::now();
        // h_flakes ist bereits aktuell – kein Memcpy nötig
        auto t2=std::chrono::high_resolution_clock::now();
        kMs=(float)std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count()/1000.f;
        fMs=(float)std::chrono::duration_cast<std::chrono::microseconds>(t2-t0).count()/1000.f;
    } else {
        // GPU-Pfad: CUDA-Events
        int blocks=(g_numFlakes+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
        cudaEventRecord(g_evFrameStart);
        cudaEventRecord(g_evKernelStart);
        updateFlakesKernel<<<blocks,THREADS_PER_BLOCK>>>(d_flakes,d_rng,g_numFlakes,DT,WIND_BASE);
        cudaEventRecord(g_evKernelStop);
        cudaMemcpy(h_flakes,d_flakes,g_numFlakes*sizeof(Flake),cudaMemcpyDeviceToHost);
        cudaEventRecord(g_evFrameStop);
        cudaEventSynchronize(g_evFrameStop);
        cudaEventElapsedTime(&kMs,g_evKernelStart,g_evKernelStop);
        cudaEventElapsedTime(&fMs,g_evFrameStart, g_evFrameStop);
    }

    // 1-Sekunden-Akkumulation
    static int lastTime=0;
    int now=glutGet(GLUT_ELAPSED_TIME);
    int dtMs=now-lastTime; lastTime=now;
    g_kernelMsAcc+=kMs; g_frameMsAcc+=fMs;
    g_perfFrameAcc++; g_perfTimeAcc+=dtMs;
    if(g_perfTimeAcc>=1000){
        float n=(float)g_perfFrameAcc;
        g_fps=n*1000.f/(float)g_perfTimeAcc;
        g_kernelMs=g_kernelMsAcc/n;
        g_frameMs =g_frameMsAcc /n;
        g_kernelMsAcc=0.f;g_frameMsAcc=0.f;
        g_perfFrameAcc=0;g_perfTimeAcc=0;
    }

    glClear(GL_COLOR_BUFFER_BIT);
    setSceneProj();
    drawScene();
    drawMenu();
    glutSwapBuffers();
    g_frame++;
}

static void idle(){glutPostRedisplay();}

static void mouse(int button,int state,int mx,int my)
{
    if(button!=GLUT_LEFT_BUTTON||state!=GLUT_DOWN)return;
    float fx=(float)mx,fy=(float)my;

    // FPS Uncap
    if(inSwitch(fx,fy,SW_UNCAP_CX,SW_UNCAP_CY,18.f,8.f)){
        g_uncapFps=!g_uncapFps; applyVsync(); return;
    }
    // CPU/GPU Switch – Klick nur wenn nicht gelockt
    if(inSwitch(fx,fy,SW_CPUGPU_CX,SW_CPUGPU_CY,18.f,9.f)){
        if(glutGet(GLUT_ELAPSED_TIME) >= g_cpuGpuLockUntil){
            g_useCPU=!g_useCPU;
            g_cpuGpuLockUntil=glutGet(GLUT_ELAPSED_TIME)+1000; // 1 Sekunde Lock
            doRestart();
        }
        return;
    }
    // Stufen
    for(int i=0;i<NUM_FLAKE_STEPS;i++) if(inStep(fx,fy,i)){g_flakeStepIdx=i;return;}
    // Restart
    if(inRect(fx,fy,BTN_X0,BTN_Y0,BTN_X1,BTN_Y1)){doRestart();return;}
}

static void keyboard(unsigned char key,int,int)
{
    if(key==27||key=='q'){
        if(d_flakes)cudaFree(d_flakes);
        if(d_rng)cudaFree(d_rng);
        delete[]h_flakes;
        if(h_rngStates)delete[]h_rngStates;
        exit(0);
    }
    if(key=='r') doRestart();
}

static void reshape(int w,int h){glViewport(0,0,w,h);}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc,char**argv)
{
    int dc=0; cudaGetDeviceCount(&dc);
    if(dc==0){fprintf(stderr,"Kein CUDA-Gerat!\n");return 1;}
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop,0);
    printf("CUDA-Gerat: %s\n",prop.name);

    glutInit(&argc,argv);
    glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGBA);
    glutInitWindowSize(WIN_W,WIN_H);
    glutCreateWindow("CUDA Schnee-Simulation");

    GLenum err=glewInit();
    if(err!=GLEW_OK){fprintf(stderr,"GLEW: %s\n",glewGetErrorString(err));return 1;}

#ifdef _WIN32
    g_wglSwapInterval=(PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT");
    if(g_wglSwapInterval){g_wglSwapInterval(1);printf("VSync: an (Toggle im Menu)\n");}
    else printf("wglSwapIntervalEXT nicht verfugbar\n");
#endif

    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

    allocAndInit();   // startet im GPU-Modus

    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutReshapeFunc(reshape);

    printf("ESC/q=Beenden | r=Restart | Klick auf Menu\n");
    glutMainLoop();
    return 0;
}
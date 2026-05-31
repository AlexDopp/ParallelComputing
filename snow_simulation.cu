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
 * Abhängigkeiten:
 *   sudo apt install freeglut3-dev libglew-dev
 *
 * Steuerung:
 *   ESC / q  – Beenden
 *   r        – Simulation zurücksetzen
 */

#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <GL/freeglut.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <curand_kernel.h>

//  Konfiguration
static constexpr int   WIN_W             = 900;
static constexpr int   WIN_H             = 700;
static constexpr int   NUM_FLAKES        = 4096;
static constexpr int   THREADS_PER_BLOCK = 128;

static constexpr float DT           = 0.002f;
static constexpr float GRAVITY      = 0.12f;
static constexpr float WIND_BASE    = 0.120f;  // 4x ursprünglich
static constexpr float FLAKE_RADIUS = 0.004f;

//  Szenen-Geometrie (normalisiert [0,1])
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
    { TREE_X - 0.10f,  TREE_X + 0.10f,  GROUND_Y + 0.06f, GROUND_Y + 0.22f },
    { TREE_X - 0.075f, TREE_X + 0.075f, GROUND_Y + 0.17f, GROUND_Y + 0.33f },
    { TREE_X - 0.05f,  TREE_X + 0.05f,  GROUND_Y + 0.28f, GROUND_Y + 0.44f },
};

__device__ __constant__ TriLayer TREE_LAYERS[3] = {
    { TREE_X - 0.10f,  TREE_X + 0.10f,  GROUND_Y + 0.06f, GROUND_Y + 0.22f },
    { TREE_X - 0.075f, TREE_X + 0.075f, GROUND_Y + 0.17f, GROUND_Y + 0.33f },
    { TREE_X - 0.05f,  TREE_X + 0.05f,  GROUND_Y + 0.28f, GROUND_Y + 0.44f },
};

struct Flake {
    float x, y;
    float vx, vy;
    float radius;
    float spawnDelay;
    float fallSpeed;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Device-Hilfsfunktionen
// ─────────────────────────────────────────────────────────────────────────────

__device__ float pointSegmentDist(float px, float py,
                                   float ax, float ay,
                                   float bx, float by,
                                   float &nx, float &ny)
{
    float dx = bx - ax, dy = by - ay;
    float len2 = dx*dx + dy*dy;
    float t = 0.f;
    if (len2 > 1e-9f)
        t = fminf(1.f, fmaxf(0.f, ((px-ax)*dx + (py-ay)*dy) / len2));
    float cx = ax + t*dx, cy = ay + t*dy;
    float ex = px - cx, ey = py - cy;
    float d  = sqrtf(ex*ex + ey*ey);
    if (d > 1e-9f) { nx = ex/d; ny = ey/d; }
    else            { nx = 0.f; ny = 1.f;  }
    return d;
}

__device__ bool collideRoof(float x, float y, float r,
                             float lx, float ly,
                             float rx2, float ry,
                             float px, float py,
                             float &nx, float &ny, float &pen)
{
    float n1x,n1y, d1 = pointSegmentDist(x,y, lx,ly,  px,py,  n1x,n1y);
    float n2x,n2y, d2 = pointSegmentDist(x,y, rx2,ry, px,py,  n2x,n2y);
    float n3x,n3y, d3 = pointSegmentDist(x,y, lx,ly,  rx2,ry, n3x,n3y);

    float d = fminf(fminf(d1,d2),d3);
    if (d < r) {
        pen = r - d;
        if      (d <= d2 && d <= d3) { nx = n1x; ny = n1y; }
        else if (d2 <= d1 && d2 <= d3){ nx = n2x; ny = n2y; }
        else                           { nx = n3x; ny = n3y; }
        return true;
    }
    return false;
}

__device__ bool collideAABB(float x, float y, float r,
                              float bx0, float by0, float bx1, float by1,
                              float &nx, float &ny, float &pen)
{
    float cx   = fmaxf(bx0, fminf(x, bx1));
    float cy   = fmaxf(by0, fminf(y, by1));
    float dx   = x - cx, dy = y - cy;
    float dist = sqrtf(dx*dx + dy*dy);

    if (dist < 1e-9f) {
        float dl = x-bx0, dr = bx1-x, db = y-by0, dt = by1-y;
        float mn = fminf(fminf(dl,dr),fminf(db,dt));
        if      (mn==dl){ nx=-1.f; ny= 0.f; pen=r+dl; }
        else if (mn==dr){ nx= 1.f; ny= 0.f; pen=r+dr; }
        else if (mn==db){ nx= 0.f; ny=-1.f; pen=r+db; }
        else             { nx= 0.f; ny= 1.f; pen=r+dt; }
        return true;
    }
    if (dist < r) {
        pen = r - dist;
        nx = dx/dist; ny = dy/dist;
        return true;
    }
    return false;
}

// Punkt-in-Dreieck (Vorzeichen der Kreuzprodukte)
__device__ bool pointInTriangle(float px, float py,
                                 float ax, float ay,
                                 float bx, float by,
                                 float cx, float cy)
{
    float d1 = (px-bx)*(ay-by) - (ax-bx)*(py-by);
    float d2 = (px-cx)*(by-cy) - (bx-cx)*(py-cy);
    float d3 = (px-ax)*(cy-ay) - (cx-ax)*(py-ay);
    bool has_neg = (d1<0)||(d2<0)||(d3<0);
    bool has_pos = (d1>0)||(d2>0)||(d3>0);
    return !(has_neg && has_pos);
}

// Prüft ob Punkt innerhalb des Hausdach-Dreiecks liegt.
// Verwendet die ECHTE Basislinie (HOUSE_WALL_Y1) ohne margin,
// damit Flocken die auf der Basislinie sitzen ebenfalls erfasst werden.
__device__ bool insideHouseRoof(float x, float y)
{
    // Schrägseiten: leicht nach innen gezogen (margin auf X/Spitze)
    // Basis: kein margin nach unten – HOUSE_WALL_Y1 ist harte Grenze
    float margin = 0.003f;
    return pointInTriangle(x, y,
                           HOUSE_X0 + margin, HOUSE_WALL_Y1,  // linke Ecke: Y bleibt exakt
                           HOUSE_X1 - margin, HOUSE_WALL_Y1,  // rechte Ecke: Y bleibt exakt
                           ROOF_PEAK_X, ROOF_PEAK_Y - margin);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Kernel
// ─────────────────────────────────────────────────────────────────────────────

__global__ void initRngKernel(curandState* rngStates, int n, unsigned long long seed)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    curand_init(seed + id * 6364136223846793005ULL, id, 0, &rngStates[id]);
}

__global__ void initFlakesKernel(Flake* flakes, curandState* rngStates, int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    curandState& rng = rngStates[id];

    int step = (int)(curand_uniform(&rng) * 10.9999f);
    float fs = 1.0f + step * 0.05f;

    flakes[id].x          = curand_uniform(&rng);
    flakes[id].y          = curand_uniform(&rng);
    flakes[id].vx         = (curand_uniform(&rng) - 0.5f) * 0.005f;
    flakes[id].vy         = -(curand_uniform(&rng) * 0.002f + 0.0015f);
    flakes[id].radius     = FLAKE_RADIUS * (0.6f + curand_uniform(&rng) * 0.8f);
    flakes[id].spawnDelay = 0.f;
    flakes[id].fallSpeed  = fs;
}

__global__ void updateFlakesKernel(Flake* flakes, curandState* rngStates, int n,
                                    float dt, float wind)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;

    Flake f = flakes[id];
    curandState& rng = rngStates[id];

    if (f.x < -5.f) return;

    if (f.spawnDelay > 0.f) {
        f.spawnDelay -= dt;
        flakes[id] = f;
        return;
    }

    float personalWind = wind * 0.1f + (curand_uniform(&rng) - 0.5f) * wind * 4.0f;
    float turbX = (curand_uniform(&rng) - 0.5f) * 0.003f;
    float turbY = (curand_uniform(&rng) - 0.5f) * 0.001f;

    f.vx += (personalWind + turbX) * dt;
    f.vy -= GRAVITY * f.fallSpeed * dt + turbY * dt;

    f.vx *= 0.98f;
    f.vy *= 0.99f;

    float maxVy = -0.003f * f.fallSpeed;
    if (f.vy < maxVy) f.vy = maxVy;
    if (f.vx >  0.004f) f.vx =  0.004f;
    if (f.vx < -0.004f) f.vx = -0.004f;

    f.x += f.vx;
    f.y += f.vy;

    if (f.x < 0.f) f.x += 1.f;
    if (f.x > 1.f) f.x -= 1.f;

    auto respawn = [&]() {
        f.x          = curand_uniform(&rng);
        f.y          = 1.02f;
        f.vx         = (curand_uniform(&rng) - 0.5f) * 0.005f;
        f.vy         = -(curand_uniform(&rng) * 0.002f + 0.0015f);
        f.spawnDelay = curand_uniform(&rng) * 1.0f;
        int step     = (int)(curand_uniform(&rng) * 10.9999f);
        f.fallSpeed  = 1.0f + step * 0.05f;
    };

    if (f.y < -0.02f) {
        if (curand_uniform(&rng) < 0.010f) {
            f.x = -10.f; f.y = -10.f;
            f.vx = 0.f;  f.vy = 0.f;
            f.spawnDelay = 0.f;
            flakes[id] = f;
            return;
        }
        respawn();
        flakes[id] = f;
        return;
    }

    // Kollisionsauflösung (3 Iterationen)
    float nx, ny, pen;
    float restitution = 0.05f;

    auto resolveCollision = [&](float cnx, float cny, float cpen) {
        f.x += cnx * cpen;
        f.y += cny * cpen;
        float vn = f.vx * cnx + f.vy * cny;
        if (vn < 0.f) {
            f.vx -= (1.f + restitution) * vn * cnx;
            f.vy -= (1.f + restitution) * vn * cny;
        }
    };

    for (int iter = 0; iter < 3; iter++) {
        if (collideAABB(f.x, f.y, f.radius,
                        HOUSE_X0, HOUSE_WALL_Y0, HOUSE_X1, HOUSE_WALL_Y1,
                        nx, ny, pen))
            resolveCollision(nx, ny, pen);

        if (collideRoof(f.x, f.y, f.radius,
                        HOUSE_X0, HOUSE_WALL_Y1,
                        HOUSE_X1, HOUSE_WALL_Y1,
                        ROOF_PEAK_X, ROOF_PEAK_Y,
                        nx, ny, pen))
            resolveCollision(nx, ny, pen);

        if (collideAABB(f.x, f.y, f.radius,
                        TREE_TRUNK_X0, TREE_TRUNK_Y0, TREE_TRUNK_X1, TREE_TRUNK_Y1,
                        nx, ny, pen))
            resolveCollision(nx, ny, pen);

        for (int i = 0; i < 3; i++) {
            const TriLayer& L = TREE_LAYERS[i];
            if (collideRoof(f.x, f.y, f.radius,
                            L.x0, L.yBase, L.x1, L.yBase,
                            TREE_X, L.yTip,
                            nx, ny, pen))
                resolveCollision(nx, ny, pen);
        }
    }

    // ── Sanitizer Hausdach ───────────────────────────────────────────────────
    // insideHouseRoof: Basis-Y ohne margin → erwischt auch Flocken auf der Basislinie
    if (insideHouseRoof(f.x, f.y)) {
        respawn();
        flakes[id] = f;
        return;
    }

    // ── Sanitizer Baumetagen ─────────────────────────────────────────────────
    for (int i = 0; i < 3; i++) {
        const TriLayer& L = TREE_LAYERS[i];
        float margin = 0.004f;
        if (pointInTriangle(f.x, f.y,
                            L.x0 + margin, L.yBase,        // Basis-Y ohne margin
                            L.x1 - margin, L.yBase,
                            TREE_X, L.yTip - margin)) {
            respawn();
            flakes[id] = f;
            return;
        }
    }

    flakes[id] = f;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Host-Zustand
// ─────────────────────────────────────────────────────────────────────────────

static Flake*       d_flakes = nullptr;
static Flake*       h_flakes = nullptr;
static curandState* d_rng    = nullptr;
static int          g_frame  = 0;
static float        g_wind   = WIND_BASE;

static void cudaInit()
{
    cudaMalloc(&d_flakes, NUM_FLAKES * sizeof(Flake));
    cudaMalloc(&d_rng,    NUM_FLAKES * sizeof(curandState));
    h_flakes = new Flake[NUM_FLAKES];

    int blocks = (NUM_FLAKES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    initRngKernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_rng, NUM_FLAKES, (unsigned long long)time(nullptr));
    cudaDeviceSynchronize();
    initFlakesKernel<<<blocks, THREADS_PER_BLOCK>>>(d_flakes, d_rng, NUM_FLAKES);
    cudaDeviceSynchronize();
}

// ─────────────────────────────────────────────────────────────────────────────
//  OpenGL  (Neon-Glas-Stil)
// ─────────────────────────────────────────────────────────────────────────────

static void glowLine(float x0, float y0, float x1, float y1,
                     float r, float g, float b)
{
    glLineWidth(6.0f);
    glColor4f(r, g, b, 0.08f);
    glBegin(GL_LINES); glVertex2f(x0,y0); glVertex2f(x1,y1); glEnd();

    glLineWidth(3.5f);
    glColor4f(r, g, b, 0.20f);
    glBegin(GL_LINES); glVertex2f(x0,y0); glVertex2f(x1,y1); glEnd();

    glLineWidth(1.8f);
    glColor4f(r, g, b, 0.55f);
    glBegin(GL_LINES); glVertex2f(x0,y0); glVertex2f(x1,y1); glEnd();

    glLineWidth(0.8f);
    glColor4f(1.0f, 1.0f, 1.0f, 0.90f);
    glBegin(GL_LINES); glVertex2f(x0,y0); glVertex2f(x1,y1); glEnd();
}

static void glowRect(float x0, float y0, float x1, float y1,
                     float r, float g, float b)
{
    glowLine(x0,y0, x1,y0, r,g,b);
    glowLine(x1,y0, x1,y1, r,g,b);
    glowLine(x1,y1, x0,y1, r,g,b);
    glowLine(x0,y1, x0,y0, r,g,b);
}

static void glowTriangle(float ax, float ay, float bx, float by,
                          float cx, float cy,
                          float r, float g, float b)
{
    glowLine(ax,ay, bx,by, r,g,b);
    glowLine(bx,by, cx,cy, r,g,b);
    glowLine(cx,cy, ax,ay, r,g,b);
}

static void drawScene()
{
    // Schwarzer Hintergrund
    glColor4f(0.f, 0.f, 0.f, 1.f);
    glBegin(GL_QUADS);
    glVertex2f(0,0); glVertex2f(1,0);
    glVertex2f(1,1); glVertex2f(0,1);
    glEnd();

    // Haus – Wand (Amber)
    glowRect(HOUSE_X0, HOUSE_WALL_Y0, HOUSE_X1, HOUSE_WALL_Y1,
             1.0f, 0.55f, 0.05f);
    // Fenster links
    glowRect(HOUSE_X0+0.04f, HOUSE_WALL_Y0+0.10f,
             HOUSE_X0+0.10f, HOUSE_WALL_Y0+0.18f,
             1.0f, 0.90f, 0.20f);
    // Fenster rechts
    glowRect(HOUSE_X1-0.10f, HOUSE_WALL_Y0+0.10f,
             HOUSE_X1-0.04f, HOUSE_WALL_Y0+0.18f,
             1.0f, 0.90f, 0.20f);
    // Tür (Magenta)
    glowRect(ROOF_PEAK_X-0.035f, HOUSE_WALL_Y0,
             ROOF_PEAK_X+0.035f, HOUSE_WALL_Y0+0.12f,
             0.85f, 0.10f, 0.70f);
    // Dach (Rot)
    glowTriangle(HOUSE_X0, HOUSE_WALL_Y1,
                 HOUSE_X1, HOUSE_WALL_Y1,
                 ROOF_PEAK_X, ROOF_PEAK_Y,
                 1.0f, 0.15f, 0.15f);

    // Baum – Stamm (Orange-Braun)
    glowRect(TREE_TRUNK_X0, TREE_TRUNK_Y0, TREE_TRUNK_X1, TREE_TRUNK_Y1,
             0.80f, 0.40f, 0.05f);
    // Etagen (Grüntöne)
    static const float treeCol[3][3] = {
        {0.10f, 1.00f, 0.35f},
        {0.15f, 0.85f, 0.50f},
        {0.20f, 0.70f, 0.65f}
    };
    for (int i = 0; i < 3; i++) {
        const TriLayer& L = TREE_LAYERS_HOST[i];
        glowTriangle(L.x0, L.yBase, L.x1, L.yBase, TREE_X, L.yTip,
                     treeCol[i][0], treeCol[i][1], treeCol[i][2]);
    }

    // Schneeflocken – drei Passes für Glow-Effekt
    glEnable(GL_POINT_SMOOTH);

    glPointSize(7.0f);
    glBegin(GL_POINTS);
    for (int i = 0; i < NUM_FLAKES; i++) {
        const Flake& fl = h_flakes[i];
        if (fl.x < -5.f || fl.spawnDelay > 0.f) continue;
        glColor4f(0.7f, 0.92f, 1.0f, 0.12f);
        glVertex2f(fl.x, fl.y);
    }
    glEnd();

    glPointSize(4.0f);
    glBegin(GL_POINTS);
    for (int i = 0; i < NUM_FLAKES; i++) {
        const Flake& fl = h_flakes[i];
        if (fl.x < -5.f || fl.spawnDelay > 0.f) continue;
        glColor4f(0.8f, 0.95f, 1.0f, 0.35f);
        glVertex2f(fl.x, fl.y);
    }
    glEnd();

    glPointSize(2.0f);
    glBegin(GL_POINTS);
    for (int i = 0; i < NUM_FLAKES; i++) {
        const Flake& fl = h_flakes[i];
        if (fl.x < -5.f || fl.spawnDelay > 0.f) continue;
        glColor4f(1.0f, 1.0f, 1.0f, 0.95f);
        glVertex2f(fl.x, fl.y);
    }
    glEnd();
}

// ─────────────────────────────────────────────────────────────────────────────
//  GLUT-Callbacks
// ─────────────────────────────────────────────────────────────────────────────

static void display()
{
    int blocks = (NUM_FLAKES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    g_wind = WIND_BASE;

    updateFlakesKernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_flakes, d_rng, NUM_FLAKES, DT, g_wind);
    cudaDeviceSynchronize();

    cudaMemcpy(h_flakes, d_flakes, NUM_FLAKES * sizeof(Flake), cudaMemcpyDeviceToHost);

    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION); glLoadIdentity();
    glOrtho(0, 1, 0, 1, -1, 1);
    glMatrixMode(GL_MODELVIEW);  glLoadIdentity();

    drawScene();
    glutSwapBuffers();
    g_frame++;
}

static void idle() { glutPostRedisplay(); }

static void keyboard(unsigned char key, int, int)
{
    if (key == 27 || key == 'q') {
        cudaFree(d_flakes); cudaFree(d_rng); delete[] h_flakes; exit(0);
    }
    if (key == 'r') {
        g_frame = 0;
        int blocks = (NUM_FLAKES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        initRngKernel<<<blocks, THREADS_PER_BLOCK>>>(
            d_rng, NUM_FLAKES, (unsigned long long)time(nullptr));
        cudaDeviceSynchronize();
        initFlakesKernel<<<blocks, THREADS_PER_BLOCK>>>(d_flakes, d_rng, NUM_FLAKES);
        cudaDeviceSynchronize();
    }
}

static void reshape(int w, int h) { glViewport(0, 0, w, h); }

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "Kein CUDA-fähiges Gerät gefunden!\n");
        return 1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("CUDA-Gerät: %s\n", prop.name);
    printf("Schneeflocken: %d   |   Threads/Block: %d   |   Blöcke: %d\n",
           NUM_FLAKES, THREADS_PER_BLOCK, NUM_FLAKES / THREADS_PER_BLOCK);

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(WIN_W, WIN_H);
    glutCreateWindow("CUDA Schnee-Simulation");

    GLenum glewErr = glewInit();
    if (glewErr != GLEW_OK) {
        fprintf(stderr, "GLEW-Fehler: %s\n", glewGetErrorString(glewErr));
        return 1;
    }

    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    cudaInit();

    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);

    printf("Steuerung: ESC/q = Beenden  |  r = Reset\n");
    glutMainLoop();
    return 0;
}

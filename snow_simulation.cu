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
static constexpr int   WIN_W          = 900;
static constexpr int   WIN_H          = 700;
static constexpr int   NUM_FLAKES     = 4096;   // muss Vielfaches von THREADS_PER_BLOCK sein
static constexpr int   THREADS_PER_BLOCK = 128; // 1 Thread = 1 Schneeflocke → 64 Blöcke

// Koordinatensystem: [0,1] x [0,1], Ursprung links-unten
static constexpr float DT            = 0.003f;  // Zeitschritt
static constexpr float GRAVITY       = 0.12f;   // Fallbeschleunigung
static constexpr float WIND_BASE     = 0.030f;  // Basis-Windstärke (nach rechts)
static constexpr float FLAKE_RADIUS  = 0.004f;  // Kollisionsradius

//  Szenen-Geometrie (normalisiert [0,1])

// Boden
static constexpr float GROUND_Y      = 0.12f;

// Haus: achsparalleles Rechteck (Wand) + Dreieck (Dach)
static constexpr float HOUSE_X0      = 0.42f;
static constexpr float HOUSE_X1      = 0.74f;
static constexpr float HOUSE_WALL_Y0 = GROUND_Y;
static constexpr float HOUSE_WALL_Y1 = 0.48f;
// Dach: Spitze
static constexpr float ROOF_PEAK_X   = (HOUSE_X0 + HOUSE_X1) * 0.5f;
static constexpr float ROOF_PEAK_Y   = 0.70f;

// Baum (links): Stamm + 3 dreieckige Etagen
static constexpr float TREE_X        = 0.18f;   // Mittelachse
static constexpr float TREE_TRUNK_X0 = TREE_X - 0.018f;
static constexpr float TREE_TRUNK_X1 = TREE_X + 0.018f;
static constexpr float TREE_TRUNK_Y0 = GROUND_Y;
static constexpr float TREE_TRUNK_Y1 = GROUND_Y + 0.10f;
// Etagen (von unten nach oben)
struct TriLayer { float x0, x1, yBase, yTip; };

// Host-Version (für drawScene)
static constexpr TriLayer TREE_LAYERS_HOST[3] = {
    { TREE_X - 0.10f,  TREE_X + 0.10f,  GROUND_Y + 0.06f, GROUND_Y + 0.22f },
    { TREE_X - 0.075f, TREE_X + 0.075f, GROUND_Y + 0.17f, GROUND_Y + 0.33f },
    { TREE_X - 0.05f,  TREE_X + 0.05f,  GROUND_Y + 0.28f, GROUND_Y + 0.44f },
};

// Device-Version (für Kernel)
__device__ __constant__ TriLayer TREE_LAYERS[3] = {
    { TREE_X - 0.10f,  TREE_X + 0.10f,  GROUND_Y + 0.06f, GROUND_Y + 0.22f },
    { TREE_X - 0.075f, TREE_X + 0.075f, GROUND_Y + 0.17f, GROUND_Y + 0.33f },
    { TREE_X - 0.05f,  TREE_X + 0.05f,  GROUND_Y + 0.28f, GROUND_Y + 0.44f },
};

//  Schneeflocken-Zustand (auf dem Device)
struct Flake {
    float x, y;          // Position
    float vx, vy;        // Geschwindigkeit
    float radius;        // individueller Radius
    float spawnDelay;    // verbleibende Wartezeit in Sekunden (0 = aktiv)
};

//  Device-Hilfsfunktionen: Kollision

// Kürzeste Distanz Punkt → Strecke
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
    float d = sqrtf(ex*ex + ey*ey);
    if (d > 1e-9f) { nx = ex/d; ny = ey/d; }
    else            { nx = 0.f; ny = 1.f;  }
    return d;
}

// Kollision mit Dreieck-Dachkante (zwei Seiten)
__device__ bool collideRoof(float x, float y, float r,
                             float lx, float ly,   // linke Ecke
                             float rx2, float ry,   // rechte Ecke
                             float px, float py,   // Spitze
                             float &nx, float &ny, float &pen)
{
    // Linke Dachseite
    float n1x, n1y, d1 = pointSegmentDist(x,y, lx,ly, px,py, n1x,n1y);
    // Rechte Dachseite
    float n2x, n2y, d2 = pointSegmentDist(x,y, rx2,ry, px,py, n2x,n2y);

    float d = fminf(d1, d2);
    if (d < r) {
        pen = r - d;
        if (d1 < d2) { nx = n1x; ny = n1y; }
        else          { nx = n2x; ny = n2y; }
        return true;
    }
    return false;
}

// Kollision mit achsparallelem Rechteck
__device__ bool collideAABB(float x, float y, float r,
                              float bx0, float by0, float bx1, float by1,
                              float &nx, float &ny, float &pen)
{
    // Nächster Punkt im Rechteck
    float cx = fmaxf(bx0, fminf(x, bx1));
    float cy = fmaxf(by0, fminf(y, by1));
    float dx = x - cx, dy = y - cy;
    float dist = sqrtf(dx*dx + dy*dy);

    // Punkt liegt innerhalb des Rechtecks
    if (dist < 1e-9f) {
        // Stoß in der Richtung mit geringstem Eindringen
        float dl = x - bx0, dr = bx1 - x, db = y - by0, dt = by1 - y;
        float mn = fminf(fminf(dl, dr), fminf(db, dt));
        if      (mn == dl) { nx = -1.f; ny = 0.f; pen = r + dl; }
        else if (mn == dr) { nx =  1.f; ny = 0.f; pen = r + dr; }
        else if (mn == db) { nx = 0.f; ny = -1.f; pen = r + db; }
        else               { nx = 0.f; ny =  1.f; pen = r + dt; }
        return true;
    }
    if (dist < r) {
        pen = r - dist;
        nx = dx / dist; ny = dy / dist;
        return true;
    }
    return false;
}

//  RNG-Initialisierungs-Kernel
__global__ void initRngKernel(curandState* rngStates, int n, unsigned long long seed)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    // Jede Flocke bekommt einen eigenen, unabhängigen RNG-Stream
    curand_init(seed + id * 6364136223846793005ULL, id, 0, &rngStates[id]);
}

//  Initialisierungs-Kernel
__global__ void initFlakesKernel(Flake* flakes, curandState* rngStates, int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;

    curandState& rng = rngStates[id];

    flakes[id].x          = curand_uniform(&rng);
    flakes[id].y          = curand_uniform(&rng);
    flakes[id].vx         = (curand_uniform(&rng) - 0.5f) * 0.005f;
    flakes[id].vy         = -(curand_uniform(&rng) * 0.002f + 0.0015f);
    flakes[id].radius     = FLAKE_RADIUS * (0.6f + curand_uniform(&rng) * 0.8f);
    flakes[id].spawnDelay = 0.f;
}

//  Update-Kernel  (1 Thread pro Schneeflocke)
__global__ void updateFlakesKernel(Flake* flakes, curandState* rngStates, int n,
                                    float dt, float wind)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;

    Flake f = flakes[id];
    curandState& rng = rngStates[id];

    // Permanent verschwundene Flocken überspringen
    if (f.x < -5.f) return;

    // Delay runterzählen - Flocke wartet unsichtbar oberhalb des Bildschirms
    if (f.spawnDelay > 0.f) {
        f.spawnDelay -= dt;
        flakes[id] = f;
        return;
    }

    // Globaler Wind sehr schwach halten, individuelle Variation pro Flocke
    float personalWind = wind * 0.1f + (curand_uniform(&rng) - 0.5f) * wind * 4.0f;
    float turbX = (curand_uniform(&rng) - 0.5f) * 0.003f;
    float turbY = (curand_uniform(&rng) - 0.5f) * 0.001f;

    f.vx += (personalWind + turbX) * dt;
    f.vy -= GRAVITY * dt + turbY * dt;

    // Dämpfung (Luftwiderstand)
    f.vx *= 0.98f;
    f.vy *= 0.99f;

    // Maximale Fallgeschwindigkeit begrenzen
    if (f.vy < -0.004f) f.vy = -0.004f;
    if (f.vx >  0.005f) f.vx =  0.005f;
    if (f.vx < -0.005f) f.vx = -0.005f;

    f.x += f.vx;
    f.y += f.vy;

    // Horizontaler Wrap-around
    if (f.x < 0.f) f.x += 1.f;
    if (f.x > 1.f) f.x -= 1.f;

    // Unten raus → mit kleiner Chance nicht neu spawnen
    if (f.y < -0.02f) {
        if (curand_uniform(&rng) < 0.010f) {
            // Flocke permanent parken - weit außerhalb des sichtbaren Bereichs
            f.x = -10.f; f.y = -10.f;
            f.vx = 0.f;  f.vy = 0.f;
            f.spawnDelay = 0.f;
            flakes[id] = f;
            return;
        }
        f.x          = curand_uniform(&rng);
        f.y          = 1.02f;
        f.vx         = (curand_uniform(&rng) - 0.5f) * 0.005f;
        f.vy         = -(curand_uniform(&rng) * 0.002f + 0.0015f);
        f.spawnDelay = curand_uniform(&rng) * 1.0f;
    }

    // Kollisionsvariablen
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

    // Haus: Wand (AABB)
    if (collideAABB(f.x, f.y, f.radius,
                    HOUSE_X0, HOUSE_WALL_Y0, HOUSE_X1, HOUSE_WALL_Y1,
                    nx, ny, pen))
        resolveCollision(nx, ny, pen);

    // Hausdach (zwei Dreiecksseiten)
    if (collideRoof(f.x, f.y, f.radius,
                    HOUSE_X0, HOUSE_WALL_Y1,
                    HOUSE_X1, HOUSE_WALL_Y1,
                    ROOF_PEAK_X, ROOF_PEAK_Y,
                    nx, ny, pen))
        resolveCollision(nx, ny, pen);

    // Baum: Stamm (AABB)
    if (collideAABB(f.x, f.y, f.radius,
                    TREE_TRUNK_X0, TREE_TRUNK_Y0, TREE_TRUNK_X1, TREE_TRUNK_Y1,
                    nx, ny, pen))
        resolveCollision(nx, ny, pen);

    // Baum: dreieckige Etagen
#pragma unroll
    for (int i = 0; i < 3; i++) {
        const TriLayer& L = TREE_LAYERS[i];
        if (collideRoof(f.x, f.y, f.radius,
                        L.x0, L.yBase, L.x1, L.yBase,
                        TREE_X, L.yTip,
                        nx, ny, pen))
            resolveCollision(nx, ny, pen);
    }

    flakes[id] = f;
}

//  Host-Zustand
static Flake*       d_flakes   = nullptr;
static Flake*       h_flakes   = nullptr;
static curandState* d_rng      = nullptr;   // persistenter RNG-State pro Flocke
static int          g_frame    = 0;
static float        g_wind     = WIND_BASE;

//  CUDA-Initialisierung
static void cudaInit()
{
    cudaMalloc(&d_flakes,  NUM_FLAKES * sizeof(Flake));
    cudaMalloc(&d_rng,     NUM_FLAKES * sizeof(curandState));
    h_flakes = new Flake[NUM_FLAKES];

    int blocks = (NUM_FLAKES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // RNG einmalig initialisieren
    initRngKernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_rng, NUM_FLAKES, (unsigned long long)time(nullptr));
    cudaDeviceSynchronize();

    // Flocken initialisieren (nutzt bereits den persistenten RNG)
    initFlakesKernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_flakes, d_rng, NUM_FLAKES);
    cudaDeviceSynchronize();
}

//  OpenGL-Zeichenroutinen

// Hilfsmakro: Farbe setzen
#define COL3(r,g,b) glColor3f((r)/255.f,(g)/255.f,(b)/255.f)

static void drawFilledTriangle(float x0,float y0,float x1,float y1,float x2,float y2)
{
    glBegin(GL_TRIANGLES);
    glVertex2f(x0,y0); glVertex2f(x1,y1); glVertex2f(x2,y2);
    glEnd();
}

static void drawFilledRect(float x0,float y0,float x1,float y1)
{
    glBegin(GL_QUADS);
    glVertex2f(x0,y0); glVertex2f(x1,y0);
    glVertex2f(x1,y1); glVertex2f(x0,y1);
    glEnd();
}

static void drawScene()
{
    // Himmel
    COL3(20,30,60);
    drawFilledRect(0,0,1,1);

    // Haus:
    // Wand
    COL3(180,140,100);
    drawFilledRect(HOUSE_X0, HOUSE_WALL_Y0, HOUSE_X1, HOUSE_WALL_Y1);

    // Fenster links
    COL3(255,220,120);
    drawFilledRect(HOUSE_X0+0.04f, HOUSE_WALL_Y0+0.10f,
                   HOUSE_X0+0.10f, HOUSE_WALL_Y0+0.18f);
    // Fenster rechts
    drawFilledRect(HOUSE_X1-0.10f, HOUSE_WALL_Y0+0.10f,
                   HOUSE_X1-0.04f, HOUSE_WALL_Y0+0.18f);
    // Tür
    COL3(100,70,40);
    drawFilledRect(ROOF_PEAK_X-0.035f, HOUSE_WALL_Y0,
                   ROOF_PEAK_X+0.035f, HOUSE_WALL_Y0+0.12f);

    // Hausdach (Dachziegel: warmes Rotbraun)
    COL3(156,75,45);
    drawFilledTriangle(HOUSE_X0, HOUSE_WALL_Y1,
                       HOUSE_X1, HOUSE_WALL_Y1,
                       ROOF_PEAK_X, ROOF_PEAK_Y);

    // Schnee auf dem Dach
    COL3(230,240,255);
    float roofOff = 0.012f;
    drawFilledTriangle(HOUSE_X0-roofOff, HOUSE_WALL_Y1,
                       HOUSE_X1+roofOff, HOUSE_WALL_Y1,
                       ROOF_PEAK_X, ROOF_PEAK_Y + 0.018f);

    // Baum
    // Stamm
    COL3(100,60,20);
    drawFilledRect(TREE_TRUNK_X0, TREE_TRUNK_Y0, TREE_TRUNK_X1, TREE_TRUNK_Y1);

    // Etagen (sattes Tannengrün)
    static const float treeGreen[3][3] = {
        {34,110,40}, {44,130,50}, {54,155,60}
    };
    for (int i = 0; i < 3; i++) {
        const TriLayer& L = TREE_LAYERS_HOST[i];
        glColor3f(treeGreen[i][0]/255.f,treeGreen[i][1]/255.f,treeGreen[i][2]/255.f);
        drawFilledTriangle(L.x0,L.yBase, L.x1,L.yBase, TREE_X,L.yTip);
        // Schnee auf jeder Etage
        COL3(210,225,255);
        float so = 0.008f;
        drawFilledTriangle(L.x0-so, L.yBase, L.x1+so, L.yBase, TREE_X, L.yTip+so*2.f);
    }

    // Schneeflocken
    glPointSize(3.5f);
    glBegin(GL_POINTS);
    for (int i = 0; i < NUM_FLAKES; i++) {
        const Flake& fl = h_flakes[i];
        glColor3f(1.f, 1.f, 1.f);
        glVertex2f(fl.x, fl.y);
    }
    glEnd();
}

//  GLUT-Callbacks
static void display()
{
    // CUDA-Update
    int blocks = (NUM_FLAKES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Wind konstant halten - Variation läuft pro Flocke über RNG
    g_wind = WIND_BASE;

    updateFlakesKernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_flakes, d_rng, NUM_FLAKES, DT, g_wind);
    cudaDeviceSynchronize();

    // Daten auf Host kopieren (für GL)
    cudaMemcpy(h_flakes, d_flakes, NUM_FLAKES * sizeof(Flake), cudaMemcpyDeviceToHost);

    // Rendering
    glClear(GL_COLOR_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    drawScene();

    glutSwapBuffers();
    g_frame++;
}

static void idle()
{
    glutPostRedisplay();
}

static void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    if (key == 27 || key == 'q') {   // ESC oder q
        cudaFree(d_flakes);
        cudaFree(d_rng);
        delete[] h_flakes;
        exit(0);
    }
    if (key == 'r') {                // Reset
        g_frame = 0;
        int blocks = (NUM_FLAKES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        initRngKernel<<<blocks, THREADS_PER_BLOCK>>>(
            d_rng, NUM_FLAKES, (unsigned long long)time(nullptr));
        cudaDeviceSynchronize();
        initFlakesKernel<<<blocks, THREADS_PER_BLOCK>>>(
            d_flakes, d_rng, NUM_FLAKES);
        cudaDeviceSynchronize();
    }
}

static void reshape(int w, int h)
{
    glViewport(0, 0, w, h);
}


int main(int argc, char** argv)
{
    // CUDA-Gerät prüfen
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

    // GLUT initialisieren
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(WIN_W, WIN_H);
    glutCreateWindow("CUDA Schnee-Simulation");

    // GLEW
    GLenum glewErr = glewInit();
    if (glewErr != GLEW_OK) {
        fprintf(stderr, "GLEW-Fehler: %s\n", glewGetErrorString(glewErr));
        return 1;
    }

    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // CUDA-Flocken initialisieren
    cudaInit();

    // GLUT-Callbacks
    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);

    printf("Steuerung: ESC/q = Beenden  |  r = Reset\n");
    glutMainLoop();

    return 0;
}

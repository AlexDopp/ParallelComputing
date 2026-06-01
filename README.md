# CUDA Snow Simulation
*Author: Alexander Doppelbauer*  

Dieses Programm simuliert fallenden Schnee in einer 2D-Szene mit Haus und Baum.  
Es dient gleichzeitig als Benchmark, um den Leistungsunterschied zwischen einer reinen CPU-Simulation  
und einer GPU-beschleunigten CUDA-Simulation zu vergleichen.  

![Video](https://github.com/user-attachments/assets/7d90e08f-8c69-4e9a-8b6f-6603b4bdde04)

## Das Koordinatensystem

Die gesamte Szene ist in einem normalisierten Koordinatensystem definiert. Beide Achsen laufen von `0.0` bis `1.0`,  
wobei `(0, 0)` die linke untere Ecke des Fensters ist und `(1, 1)` die rechte obere.  
Alle Positionen, Geschwindigkeiten und Geometriemaße sind in diesen Einheiten angegeben.  

## Die Schneeflocke: Datenstruktur

Jede Schneeflocke wird durch eine einzige Instanz der Struktur `Flake` repräsentiert:

```cpp
struct Flake {
    float x, y;          // Position in der Szene [0..1]
    float vx, vy;        // Geschwindigkeitsvektor (Einheiten pro Zeitschritt)
    float radius;        // Kollisionsradius
    float spawnDelay;    // verbleibende Wartezeit bis die Flocke aktiv wird
    float fallSpeed;     // individueller Fallgeschwindigkeits-Multiplikator
};
```

`x`, `y` beinhalten die aktuelle Position der Flocke in der Szene.  

`vx`, `vy` beinhält die Geschwindigkeit, welche zum Start zufällig aus den Bereichen [-0.0025, +0.0025], [-0.0035, -0.0015] gewählt wird.  
Danach wird die Geschwindigkeit für jedes Frame angepasst durch Physik und Dämpfung nach folgender Formeln:  
```cpp
vx += (personalWind + turbX) * dt;
vy -= GRAVITY * fallSpeed * dt + turbY * dt;
```
`radius` beinhält den Kollisionsradius  

`spawnDelay` speichert nach Erreichen von `y < 0` eine zufällige Wartezeit zwischen 0 und 1 Sekunde,  
die eine Flocke abwarten muss, bevor sie respawned.  

`fallSpeed` beinhält einen individuellen Multiplikator auf die Schwerkraft zwischen 100% und 150% in 5%-Schritten.  

## Zufallszahlengenerierung per CPU

Im CPU-Modus hat jede Flocke einen eigenen `uint64_t`-Zustandswert (`h_rngStates[id]`).  
Für jede benötigte Zufallszahl wird ein **Linear Congruential Generator (LCG)** aufgerufen:  

```cpp
static inline float lcgUniform(uint64_t& s) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((s >> 33) & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}
```

## Physik-Update pro Frame per CPU

Jedes Frame ruft das Programm cpuUpdateFlakes() auf. 
Die Funktion iteriert in einer for-Schleife sequenziell über alle Flocken und berechnet jede einzeln.    
Die folgenden Schritte werden für jede aktive Flocke in dieser Reihenfolge ausgeführt.  

### Schritt 1: Inaktive Flocken mit `spawnDelay` überspringen  
```cpp
if(f.spawnDelay > 0.f){ f.spawnDelay -= dt; continue; }
```

### Schritt 2: Wind und Turbulenz berechnen  

```cpp
personalWind = WIND_BASE * 0.1 + (random - 0.5) * WIND_BASE * 4.0;
turbX        = (random - 0.5) * 0.003;
turbY        = (random - 0.5) * 0.001;
```

Jede Flocke bekommt ihren eigenen, zufällig variierenden Windanteil.  
Das Ergebnis ist ein globaler Wind der im Mittel nach rechts weht, aber pro Flocke und Frame stark variiert.  
Zusätzlich gibt es eine kleine zufällige Turbulenz in beide Richtungen (`turbX`, `turbY`).  

### Schritt 3: Geschwindigkeit aktualisieren

```cpp
vx += (personalWind + turbX) * dt;
vy -= GRAVITY * fallSpeed * dt + turbY * dt;
```

### Schritt 4: Dämpfung (Luftwiderstand)

```cpp
vx *= 0.98;
vy *= 0.99;
```

Pro Frame wird die Geschwindigkeit um 2% (horizontal) bzw. 1% (vertikal) reduziert.  
Das modelliert den Luftwiderstand und regelt die ständige Windbeschleunigung herunter.

### Schritt 5: Geschwindigkeit begrenzen (Clamp)  

```cpp
float mvy = -0.003f * f.fallSpeed;
if(f.vy < mvy) f.vy = mvy;
if(f.vx >  0.004f) f.vx =  0.004f;
if(f.vx < -0.004f) f.vx = -0.004f;
```

Die Geschwindigkeit wird in alle Richtungen begrenzt, damit Flocken nicht beliebig schnell werden  
und bei großen Zeitschritten tief in Kollisionsgeometrie eindringen könnten.  

### Schritt 6: Position aktualisieren und horizontaler Wrap-around

```cpp
x += vx;
y += vy;
if(x < 0.0) x += 1.0;
if(x > 1.0) x -= 1.0;
```

Flocken die links aus dem Bild fliegen erscheinen rechts wieder, und umgekehrt.  


### Schritt 7: Respawn am unteren Rand

Wenn `y < -0.02` (Flocke hat den unteren Bildrand verlassen):

Flocke wird neu gespawned mit zufälliger X-Position, `y = 1.02`, neuer zufällige Geschwindigkeit,  
neuem `spawnDelay` und neuem `fallSpeed`.  

## Kollisionserkennung per CPU

Nach dem Positions-Update wird jede Flocke ein mal pro Frame gegen alle Szenen-Geometrien geprüft.  
Dabei werden zwei Typen von Kollisionsgeometrie verwendet:  

**Achsenparalleles Rechteck (AABB):** Verwendet für die Hauswand und den Baumstamm.  
Die Funktion `cpu_collideAABB` findet den nächsten Punkt auf dem Rechteck zur Flockenposition,  
berechnet den Abstand und gibt bei Überschneidung die Kollisionsnormale und die Eindringtiefe (`pen`) zurück.  

**Dreieck (Dach / Baumetagen):** Verwendet für das Hausdach und die drei Dreiecksebenen des Baums.  
Die Funktion `cpu_collideRoof` prüft alle Kanten des Dreiecks mit der Punkt-zu-Strecke-Distanzfunktion  
und verwendet die nächste Kante für die Kollisionsauflösung.  

### Kollisionsauflösung
```cpp
// Position korrigieren: Flocke entlang Normale um Eindringtiefe herausschieben
x += normalX * penetration;
y += normalY * penetration;

// Geschwindigkeit korrigieren: Anteil entlang Normale umkehren
vn = vx × normalX + vy * normalY;           // Normalanteil der Geschwindigkeit
if (vn < 0) {
    vx -= (1 + restitution) * vn * normalX;
    vy -= (1 + restitution) * vn * normalY;
}
```

Der `restitution`-Faktor von `0.05` bewirkt, dass die Flocke nach einer Kollision mit 5% der Aufprallgeschwindigkeit  
zurückprallt und somit fast vollständig inelastisch wirkt.  
Als zweite Sicherheitsebene wird nach der Kollisionsschleife geprüft, ob die Flocke trotz Auflösung noch innerhalb des Dreiecks liegt.  
Falls ja, wird die Flocke sofort neu gespawnt.  

---

## GPU-Modus: CUDA-Implementierung

Im GPU-Modus läuft die gesamte Physik-Berechnung auf der Grafikkarte. Die Logik ist dabei identisch zur CPU-Implementierung.  
Das Rendering übernimmt weiterhin die CPU.  

## Parallelisierung: Ein Thread pro Schneeflocke

Während die CPU alle Flocken sequenziell in einer `for`-Schleife berechnet, weist die GPU jedem Thread genau eine Flocke zu.  
Alle Flocken werden damit gleichzeitig und unabhängig voneinander berechnet.  

Die Threads werden in Blöcken von je 128 Threads organisiert:  
```cpp
int blocks = (g_numFlakes + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
updateFlakesKernel<<<blocks, THREADS_PER_BLOCK>>>(...);
```
Bei 4096 Flocken entstehen damit 32 Blöcke à 128 Threads. Die Flockenanzahl muss deshalb immer ein Vielfaches von 128 sein.  

## Der Kernel

Statt einer Funktion die über alle Flocken iteriert, gibt es eine `__global__`-Funktion die von der GPU für jeden Thread einmal ausgeführt wird.  

```cpp
__global__ void updateFlakesKernel(Flake* flakes, curandState* rng, int n, float dt, float wind)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= n) return;
    Flake f = flakes[id];
    ...
}
```

## Speicher

Im GPU-Modus liegen die Flocken-Daten im **Device Memory** der GPU (`d_flakes`). Der CPU-Zugriff darauf ist nicht direkt möglich.  
Nach jeder Berechnung müssen die Daten deshalb explizit zurück auf den Host kopiert werden, damit OpenGL sie rendern kann:  

```cpp
cudaMemcpy(h_flakes, d_flakes, g_numFlakes * sizeof(Flake), cudaMemcpyDeviceToHost);
```

## Zufallszahlengenerierung per GPU

Der CPU-LCG kann auf der GPU nicht verwendet werden, da jeder Thread einen eigenen unabhängigen Zufallszustand braucht der parallel verwaltet wird.  
Dafür verwendet CUDA die `curand`-Bibliothek mit einem `curandState` pro Flocke:

```cpp
curand_init(seed + id * 6364136223846793005ULL, id, 0, &rng[id]);
```

Jede Flocke bekommt damit einen eindeutigen Startzustand. Im laufenden Betrieb wird der Zustand direkt im Kernel über `curand_uniform(&r)` abgerufen.

## Geometrie im Constant Memory

Die Kollisionsgeometrie liegt auf der GPU im **Constant Memory**, also einem gecachten Speicherbereich der für Daten gedacht ist, die alle Threads gleichzeitig lesen:  

```cpp
__device__ __constant__ TriLayer TREE_LAYERS[3] = { ... };
__device__ __constant__ AABB AABB_OBJECTS[2]    = { ... };
```

Wenn alle 128 Threads eines Blocks dieselbe Adresse lesen, wird der Wert einmal geladen und per Broadcast an alle weitergegeben.  
Das ist effizienter als ein normaler Speicherzugriff bei dem jeder Thread einzeln liest.  

## Kollisionserkennung per GPU

Die Kollisionsfunktionen (`gpu_collideAABB`, `gpu_collideRoof`) sind inhaltlich identisch zur CPU-Variante,  
müssen aber als `__device__`-Funktionen deklariert werden damit der CUDA-Compiler sie für die GPU übersetzt:  

```cpp
__device__ bool gpu_collideAABB(float x, float y, float r, const AABB& b, ...)
```

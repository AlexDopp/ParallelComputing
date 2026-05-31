# CUDA Snow Simulation

## Überblick

Dieses Programm simuliert fallenden Schnee in einer 2D-Szene mit Haus und Baum.  
Es dient gleichzeitig als Benchmark, um den Leistungsunterschied zwischen einer reinen CPU-Simulation und einer GPU-beschleunigten CUDA-Simulation zu vergleichen.  

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
vx += (personalWind + turbX) × dt
vy -= GRAVITY × fallSpeed × dt + turbY × dt
```
`radius` beinhält den Kollisionsradius  

`spawnDelay` speichert nach Erreichen von `y < 0` eine zufällige Wartezeit zwischen 0 und 1 Sekunde,  
die eine Flocke abwarten muss, bevor sie respawned.  

`fallSpeed` beinhält einen individuellen Multiplikator auf die Schwerkraft zwischen 100% und 150% in 5%-Schritten.  

## Zufallszahlengenerierung im CPU-Modus

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
```
if spawnDelay > 0  →  spawnDelay -= dt, überspringen (noch nicht aktiv)
```

### Schritt 2: Wind und Turbulenz berechnen  

```
personalWind = WIND_BASE × 0.1 + (random - 0.5) × WIND_BASE × 4.0
turbX        = (random - 0.5) × 0.003
turbY        = (random - 0.5) × 0.001
```

Jede Flocke bekommt ihren eigenen, zufällig variierenden Windanteil.  
Das Ergebnis ist ein globaler Wind der im Mittel nach rechts weht, aber pro Flocke und Frame stark variiert.  
Zusätzlich gibt es eine kleine zufällige Turbulenz in beide Richtungen (`turbX`, `turbY`).  

### Schritt 3: Geschwindigkeit aktualisieren

```
vx += (personalWind + turbX) × dt
vy -= GRAVITY × fallSpeed × dt + turbY × dt
```

### Schritt 4: Dämpfung (Luftwiderstand)

```
vx *= 0.98
vy *= 0.99
```

Jedes Frame wird die Geschwindigkeit um 2% (horizontal) bzw. 1% (vertikal) reduziert.  
Das modelliert den Luftwiderstand und regelt die ständige Windbeschleunigung herunter.

### Schritt 5: Geschwindigkeit begrenzen (Clamp)  

```
vy  ≥  -0.003 × fallSpeed   (maximale Fallgeschwindigkeit)
vx  ∈  [-0.004, +0.004]     (maximale Horizontalgeschwindigkeit)
```

Die Geschwindigkeit wird nach oben und unten begrenzt, damit Flocken nicht beliebig schnell werden  
und bei großen Zeitschritten tief in Kollisionsgeometrie eindringen könnten.  

### Schritt 6: Position aktualisieren und horizontaler Wrap-around

```
x += vx
y += vy
if x < 0.0  →  x += 1.0
if x > 1.0  →  x -= 1.0
```

Flocken die links aus dem Bild fliegen erscheinen rechts wieder, und umgekehrt.  


### Schritt 8: Respawn am unteren Rand

Wenn `y < -0.02` (Flocke hat den unteren Bildrand verlassen):

Flocke wird neu gespawned mit zufälliger X-Position, `y = 1.02`, neuer zufällige Geschwindigkeit,  
neuem `spawnDelay` und neuem `fallSpeed`.  

## Kollisionserkennung

Nach dem Positions-Update wird jede Flocke ein mal pro Frame gegen alle Szenen-Geometrien geprüft.  
Dabei werden zwei Typen von Kollisionsgeometrie verwendet:  

**Achsenparalleles Rechteck (AABB):** Verwendet für die Hauswand und den Baumstamm.  
Die Funktion `cpu_collideAABB` findet den nächsten Punkt auf dem Rechteck zur Flockenposition,  
berechnet den Abstand und gibt bei Überschneidung die Kollisionsnormale und die Eindringtiefe (`pen`) zurück.  

**Dreieck (Dach / Baumetagen):** Verwendet für das Hausdach und die drei Dreiecksebenen des Baums.  
Die Funktion `cpu_collideRoof` prüft alle Kanten des Dreiecks mit der Punkt-zu-Strecke-Distanzfunktion  
und verwendet die nächste Kante für die Kollisionsauflösung.  

### Kollisionsauflösung
```
// Position korrigieren: Flocke entlang Normale um Eindringtiefe herausschieben
x += normalX × penetration
y += normalY × penetration

// Geschwindigkeit korrigieren: Anteil entlang Normale umkehren
vn = vx × normalX + vy × normalY   // Normalanteil der Geschwindigkeit
if vn < 0:
    vx -= (1 + restitution) × vn × normalX
    vy -= (1 + restitution) × vn × normalY
```

Der `restitution`-Faktor von `0.05` bewirkt, dass die Flocke nach einer Kollision mit 5% der Aufprallgeschwindigkeit  
zurückprallt und somit fast vollständig inelastisch wirkt.  
Als zweite Sicherheitsebene wird nach der Kollisionsschleife geprüft, ob die Flocke trotz Auflösung noch innerhalb einer Dreiecksgeometrie liegt.  
Falls ja, wird die Flocke sofort neu gespawnt.  

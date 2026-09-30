# -*- coding: utf-8 -*-
"""
===========================================================================
 simulator_abnahme.py — die Abnahmen des Netzlisten-Simulators
===========================================================================

 Sieben Abnahmen, jede gegen eine unabhaengige Referenz:

 1  SPANNUNGSTEILER, ideale Quelle (Ri = 0):  exakter Vergleichswert
    von Hand. Prueft Quelle (Zwangszeile) und Widerstandsstempel.

 2  RC-AUFLADUNG:  Vergleich mit der von Hand mitgerechneten
    Euler-Folge  u[k+1] = u[k] + dt*((U - u[k])/R)/C.
    Prueft Kondensatorstempel und Euler-Schritt scharf (Maschinen-
    genauigkeit), dazu die exakte Loesung als Plausibilitaet.

 3  RL-AUFLADUNG:  Vergleich mit der Euler-Folge
    i[k+1] = i[k] + dt*(U - R*i[k])/L, dazu die exakte Loesung
    i = U/R * (1 - exp(-R t/L)) als Plausibilitaet.
    Prueft Induktivitaetsstempel und Euler-Schritt.

 4  DIE B4-BRUECKE ALS NETZLISTE gegen bruecke_kern.py: 20 000 Schritte
    (0,2 s bei dt = 10 us), Schritt fuer Schritt verglichen. Der Kern
    rechnet dieselbe Methode mit fest verdrahteter Handaufloesung —
    zwei voellig getrennte Programmwege muessen dasselbe liefern.
    (Winzige Abweichungen sind erlaubt: der Simulator setzt GMIN auch
    an die Zusatzknoten q und k und loest mit Gauss statt mit der
    Kreuzregel.)

 Dazu drei Abnahmen fuer den IMPLIZITEN Euler (verfahren="implizit"):

 5  RC- und RL-AUFLADUNG gegen die implizite Folge von Hand
    (Companion-Modell aufgeloest), scharf auf Maschinengenauigkeit.

 6  STABILITAET am LC-Schwingkreis (C 1 uF mit 10 V Anfangswert,
    L 100 mH, Periode 2 ms, dt 10 us): der explizite Euler muss
    aufklingen (bekannte Schwaeche, Manuskript Teil V), der implizite
    beschraenkt bleiben. Genau dafuer ist er da.

 7  DIE BRUECKE IMPLIZIT: gleicher Endwert wie explizit bis auf
    wenige mV (beide Verfahren sind von erster Ordnung in dt).

 Aufruf:  python simulator_abnahme.py     (Programme muessen im selben
 Ordner liegen: simulator.py, bruecke_kern.py, bruecke.netz)
===========================================================================
"""
import os

import numpy as np

from simulator import Simulator, GMIN
from bruecke_kern import Bruecke

HIER = os.path.dirname(os.path.abspath(__file__))
fehler = 0


def pruefe(name, wert, schranke, einheit="V"):
    global fehler
    gut = wert < schranke
    print(f"  {'bestanden' if gut else 'DURCHGEFALLEN':13s} {name}: "
          f"groesste Abweichung {wert:.3e} {einheit} "
          f"(Schranke {schranke:.0e})")
    if not gut:
        fehler += 1


print("=" * 70)
print("  ABNAHME 1 — Spannungsteiler an idealer Quelle")
print("-" * 70)
sim = Simulator("""
V1  e 0  dc 10
R1  e m  1k
R2  m 0  2k
""", dt=1e-3)
sim.schritt()
# Vergleichswert von Hand MIT GMIN (der Simulator setzt 1e-9 S an jeden
# Knoten; ohne GMIN im Vergleichswert blieben 4,4 uV Rest stehen):
soll = (10.0 / 1000.0) / (1.0 / 1000.0 + 1.0 / 2000.0 + GMIN)
ist = sim.potential("m")
print(f"  u(m) = {ist:.12f} V   (von Hand {soll:.12f} V, "
      f"ohne GMIN {10.0*2000.0/3000.0:.12f} V)")
pruefe("Spannungsteiler", abs(ist - soll), 1e-12)

print()
print("=" * 70)
print("  ABNAHME 2 — RC-Aufladung gegen die Euler-Folge von Hand")
print("-" * 70)
U, R, C, dt, n = 10.0, 1e3, 1e-6, 1e-6, 5000
sim = Simulator(f"""
V1  e 0  dc {U}
R1  e k  {R}
C1  k 0  {C}
""", dt=dt)
# Euler-Folge von Hand MIT GMIN: der Kondensatorstrom aus der Loesung
# ist I_C = (U - u_C)/R - GMIN*u_C  (Knotensumme am Knoten k).
uC_hand = 0.0
gross = 0.0
for k in range(n):
    sim.schritt()
    uC_hand += dt * ((U - uC_hand) / R - GMIN * uC_hand) / C
    gross = max(gross, abs(sim.bauteil("C1").uC - uC_hand))
exakt = U * (1.0 - np.exp(-n * dt / (R * C)))
print(f"  u_C nach {n*dt*1e3:.0f} ms: Simulator {sim.bauteil('C1').uC:.9f} V, "
      f"Euler von Hand {uC_hand:.9f} V, exakt {exakt:.9f} V")
pruefe("RC gegen Euler-Folge", gross, 1e-12)

print()
print("=" * 70)
print("  ABNAHME 3 — RL-Aufladung gegen die Euler-Folge von Hand")
print("-" * 70)
U, R, L, dt, n = 10.0, 10.0, 0.1, 1e-6, 20000
sim = Simulator(f"""
V1  e 0  dc {U}
R1  e m  {R}
L1  m 0  {L}
""", dt=dt)
# Euler-Folge von Hand MIT GMIN: die Spulenspannung aus der Loesung ist
# u_m = (U/R - i_L) / (1/R + GMIN)  (Knotensumme am Knoten m).
iL_hand = 0.0
gross = 0.0
for k in range(n):
    sim.schritt()
    u_m = (U / R - iL_hand) / (1.0 / R + GMIN)
    iL_hand += dt * u_m / L
    gross = max(gross, abs(sim.bauteil("L1").iL - iL_hand))
exakt = U / R * (1.0 - np.exp(-R * n * dt / L))
print(f"  i_L nach {n*dt*1e3:.0f} ms: Simulator {sim.bauteil('L1').iL:.9f} A, "
      f"Euler von Hand {iL_hand:.9f} A, exakt {exakt:.9f} A")
pruefe("RL gegen Euler-Folge", gross, 1e-12, "A")

print()
print("=" * 70)
print("  ABNAHME 4 — B4-Bruecke als Netzliste gegen bruecke_kern.py")
print("-" * 70)
with open(os.path.join(HIER, "bruecke.netz")) as f:
    sim = Simulator(f.read(), dt=10e-6)
kern = Bruecke()

n = 20000
gross_uC = gross_uP = 0.0
for k in range(n):
    s = kern.schritt()
    sim.schritt()
    gross_uC = max(gross_uC, abs(sim.bauteil("C1").uC - kern.uC))
    gross_uP = max(gross_uP, abs(sim.potential("K3") - kern.uP))

print(f"  nach {n} Schritten (0,2 s):")
print(f"    u_C  Simulator {sim.bauteil('C1').uC:.9f} V   "
      f"Kern {kern.uC:.9f} V")
print(f"    u_P  Simulator {sim.potential('K3'):.9f} V   "
      f"Kern {kern.uP:.9f} V")
print(f"    Newton im Mittel: Simulator {sim.it_summe/sim.schritte:.2f}, "
      f"Kern {kern.it_summe/kern.schritte:.2f}")
pruefe("u_C gegen den Kern", gross_uC, 1e-6)
pruefe("u_P gegen den Kern", gross_uP, 1e-6)
gross_uC_ende = sim.bauteil("C1").uC        # fuer Abnahme 7

print()
print("=" * 70)
print("  ABNAHME 5 — RC und RL mit IMPLIZITEM Euler gegen die Folge von Hand")
print("-" * 70)
U, R, C, dt, n = 10.0, 1e3, 1e-6, 1e-6, 5000
sim = Simulator(f"""
V1  e 0  dc {U}
R1  e k  {R}
C1  k 0  {C}
""", dt=dt, verfahren="implizit")
# implizite Folge von Hand: Knoten k mit Companion-Modell aufgeloest:
#   u_neu = (U/R + gC*u_alt) / (1/R + GMIN + gC),  gC = C/dt
gC = C / dt
uC_hand = 0.0
gross = 0.0
for k in range(n):
    sim.schritt()
    uC_hand = (U / R + gC * uC_hand) / (1.0 / R + GMIN + gC)
    gross = max(gross, abs(sim.bauteil("C1").uC - uC_hand))
exakt = U * (1.0 - np.exp(-n * dt / (R * C)))
print(f"  u_C nach {n*dt*1e3:.0f} ms: Simulator {sim.bauteil('C1').uC:.9f} V, "
      f"implizit von Hand {uC_hand:.9f} V, exakt {exakt:.9f} V")
pruefe("RC implizit", gross, 1e-9)

U, R, L, dt, n = 10.0, 10.0, 0.1, 1e-6, 20000
sim = Simulator(f"""
V1  e 0  dc {U}
R1  e m  {R}
L1  m 0  {L}
""", dt=dt, verfahren="implizit")
# implizite Folge von Hand: Knoten m:  (u-U)/R + GMIN*u + gL*u + i_alt = 0
gL = dt / L
iL_hand = 0.0
gross = 0.0
for k in range(n):
    sim.schritt()
    u_m = (U / R - iL_hand) / (1.0 / R + GMIN + gL)
    iL_hand += gL * u_m
    gross = max(gross, abs(sim.bauteil("L1").iL - iL_hand))
exakt = U / R * (1.0 - np.exp(-R * n * dt / L))
print(f"  i_L nach {n*dt*1e3:.0f} ms: Simulator {sim.bauteil('L1').iL:.9f} A, "
      f"implizit von Hand {iL_hand:.9f} A, exakt {exakt:.9f} A")
pruefe("RL implizit", gross, 1e-9, "A")

print()
print("=" * 70)
print("  ABNAHME 6 — Stabilitaet am LC-Schwingkreis (die Nagelprobe)")
print("-" * 70)
LC_NETZ = """
C1  k 0  1u  10      * 10 V Anfangswert
L1  k 0  100m
"""
ergebnisse = {}
for verf in ("explizit", "implizit"):
    sim = Simulator(LC_NETZ, dt=10e-6, verfahren=verf)
    gross = 0.0
    for k in range(10000):                       # 0,1 s = 50 Perioden
        sim.schritt()
        gross = max(gross, abs(sim.bauteil("C1").uC))
    ergebnisse[verf] = gross
    print(f"  {verf:9s}: groesste |u_C| in 0,1 s = {gross:12.3f} V "
          f"(Anfangswert 10 V)")
if ergebnisse["explizit"] > 20.0:
    print("  bestanden      explizit klingt auf — die bekannte Schwaeche")
else:
    print("  DURCHGEFALLEN  explizit haette aufklingen muessen"); fehler += 1
if ergebnisse["implizit"] <= 10.0 + 1e-6:
    print("  bestanden      implizit bleibt beschraenkt")
else:
    print("  DURCHGEFALLEN  implizit haette beschraenkt bleiben muessen")
    fehler += 1

print()
print("=" * 70)
print("  ABNAHME 7 — die Bruecke mit implizitem Euler")
print("-" * 70)
with open(os.path.join(HIER, "bruecke.netz")) as f:
    sim = Simulator(f.read(), dt=10e-6, verfahren="implizit")
for k in range(20000):
    sim.schritt()
uC_impl = sim.bauteil("C1").uC
print(f"  u_C nach 0,2 s: implizit {uC_impl:.9f} V, "
      f"explizit (Abnahme 4) {gross_uC_ende:.9f} V")
print(f"  Newton im Mittel: {sim.it_summe/sim.schritte:.2f}")
pruefe("implizit gegen explizit", abs(uC_impl - gross_uC_ende), 5e-3)

print()
print("=" * 70)
if fehler:
    print(f"  ERGEBNIS: {fehler} Abnahme(n) DURCHGEFALLEN")
    raise SystemExit(1)
print("  ERGEBNIS: alle sieben Abnahmen bestanden")
print("=" * 70)

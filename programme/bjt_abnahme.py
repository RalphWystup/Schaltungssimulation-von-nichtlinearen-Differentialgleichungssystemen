# -*- coding: utf-8 -*-
"""
===========================================================================
 bjt_abnahme.py — die Abnahmen des Transistorstempels
===========================================================================

 Nach dem Muster von simulator_abnahme.py: jede Abnahme gegen eine
 UNABHAENGIGE Referenz, jede Zahl gerechnet oder gemessen, keine
 Behauptung ohne Beleg.

 0  DIE VIER TANGENTEN gegen den zentralen Differenzenquotienten.
    Die Ableitungen sind von Hand hergeleitet (Manuskript, Abschnitt 48);
    hier wird jede einzeln nachgemessen, ueber ein ganzes Raster von
    Arbeitspunkten und fuer alle vier Parameterkarten.

 1  DER ARBEITSPUNKT DER EMITTERSCHALTUNG gegen sein eigenes Programm
    kap08_rechnung.py (Buchkapitel 8). Zwei voellig getrennte Wege —
    dort zwei Maschengleichungen und eine 2x2-Jacobi-Matrix von Hand,
    hier eine Netzliste und die Knotenmatrix — muessen dieselben Zahlen
    liefern. Das ist die tragende Abnahme.
    Der Rest, der bleibt, ist GMIN: der Simulator setzt an jeden Knoten
    1e-9 S gegen Masse, sein Programm kennt das nicht. Deshalb wird
    ZWEIMAL verglichen: gegen sein Programm (auf gueltige Stellen) und
    gegen seine Gleichungen MIT GMIN (auf Maschinengenauigkeit).

 2  DAS AUSGANGSKENNLINIENFELD gegen die Messung Ic_Vce.txt
    (BC337, fuenf Basisstromstufen 8 ... 40 uA).

 3  EINGANGSKENNLINIE gegen Ic_Vbe.txt und STROMSTEUERKENNLINIE gegen
    Ic_Ib.txt, beide ebenfalls am BC337.

 4  EIN ZEITVERLAUF: Emitterschaltung mit Koppelkondensator und
    Signalquelle, mit Euler gerechnet. Drei Nachweise:
      a) bei abgeschalteter Quelle steht der Arbeitspunkt bis auf
         Rundungsreste still — der Koppelkondensator laedt nicht weg;
      b) der Loeser erfuellt in JEDEM Zeitschritt die Lastgerade und das
         Modell (nachgerechnet mit einer zweiten, getrennt geschriebenen
         Modellfunktion, nicht mit dem Stempel);
      c) die Grosssignalverzerrung ist sichtbar und stimmt mit der
         Exponentialfunktion ueberein: das Verhaeltnis von groesstem zu
         kleinstem Kollektorstrom ist exp(2*u_hut/(n V_T)), auf Papier
         nachrechenbar.

 5  GEGENPROBE MIT LTSPICE: seine .asc-Dateien Eigen_RW_1d/_1e enthalten
    dieselbe Fixed-Bias-Schaltung mit je einem Basiswiderstand, den er
    dort eingetragen hat. Wir rechnen mit genau diesen Widerstaenden und
    seinen .model-Karten und sagen, welches V_CE herauskommt.

 Erzeugt vier Bilder in ../Bilder/ und einen Bericht auf stdout.

 Aufruf:  python bjt_abnahme.py
===========================================================================
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulator import Simulator, MODELLE_BJT, GMIN

HIER = os.path.dirname(os.path.abspath(__file__))
BILDER = os.path.normpath(os.path.join(HIER, "..", "Bilder"))
# Die Messdateien liegen im Arbeitsbereich, zwei Ebenen darueber —
# sie gehoeren zur Transistorarbeit und werden hier nur GELESEN.
MESS = os.path.normpath(os.path.join(HIER, "..", ".."))

fehler = 0


def pruefe(name, wert, schranke, einheit=""):
    global fehler
    gut = wert < schranke
    print(f"  {'bestanden' if gut else 'DURCHGEFALLEN':13s} {name}: "
          f"{wert:.3e} {einheit} (Schranke {schranke:.0e})")
    if not gut:
        fehler += 1


def netz(datei):
    with open(os.path.join(HIER, datei)) as f:
        return f.read()


# =========================================================================
#  Messdateien des Kennlinienschreibers lesen — dieselbe Routine wie in
#  seinem bjt_extract.py, damit die Zahlen aus derselben Quelle kommen.
# =========================================================================
def lade(name):
    with open(os.path.join(MESS, name + ".txt"), encoding="utf-8-sig") as fh:
        zeilen = fh.read().splitlines()
    traces, kopf = [], None
    for i, ln in enumerate(zeilen):
        sp = ln.split("\t")
        if sp[0] == "Trace:":
            traces = [c for c in sp[1:] if c.strip()]
        if sp[0] == "Point":
            kopf = i
            break
    daten = []
    for ln in zeilen[kopf + 1:]:
        if not ln.strip():
            continue
        daten.append([float(c.strip().replace(",", ".")) if c.strip() else np.nan
                      for c in ln.split("\t")])
    a = np.array(daten, dtype=float)
    xs = [a[:, 1 + 2 * i] for i in range(len(traces))]
    ys = [a[:, 2 + 2 * i] for i in range(len(traces))]
    return traces, xs, ys


def trace_wert(label):
    return float(label.split("=")[1].replace("µA", "").replace("V", "")
                 .replace(",", ".").strip())


# =========================================================================
#  Der Messplatz: eine Netzliste, zwei verstellbare Quellen.
# =========================================================================
class Messplatz:
    """bjt_kennlinien.netz mit warmem Start: der Arbeitspunkt des
    vorigen Punktes ist der Newton-Startwert des naechsten — wie beim
    echten Sweep, und es spart die 150 gedaempften Schritte, die der
    kalte Start von u = 0 aus kostet."""

    def __init__(self, modell="BC337"):
        text = netz("bjt_kennlinien.netz")
        if modell != "BC337":
            text = text.replace("BC337", modell)
        self.sim = Simulator(text, dt=1e-3)
        self.VB = self.sim.bauteil("VB")
        self.VC = self.sim.bauteil("VC")

    def punkt(self, vb, vc):
        self.VB.U, self.VC.U = vb, vc
        self.sim.schritt()
        return self.sim.messwerte("T1")

    def bei_ib(self, ib_soll, vc, lo=0.2, hi=1.0, n=48):
        """V_B so einstellen, dass I_B seinen Sollwert hat — Bisektion um
        den Simulator herum. Dieselbe Arbeitsteilung wie in Kapitel 8.8:
        aussen die robuste Bisektion, innen der schnelle Newton."""
        for _ in range(n):
            mid = 0.5 * (lo + hi)
            if self.punkt(mid, vc)["i_b"] > ib_soll:
                hi = mid
            else:
                lo = mid
        return self.punkt(0.5 * (lo + hi), vc)


def statistik(name, rel, schranke_max, schranke_mittel):
    """relative Abweichungen -> groesste und mittlere in Prozent."""
    rel = np.asarray(rel)
    gr, mi = 100 * np.max(np.abs(rel)), 100 * np.mean(np.abs(rel))
    print(f"    {name}: groesste {gr:.2f} %, mittlere {mi:.2f} % "
          f"({len(rel)} Punkte)")
    pruefe(f"{name} groesste Abweichung", gr, schranke_max, "%")
    pruefe(f"{name} mittlere Abweichung", mi, schranke_mittel, "%")
    return gr, mi


# =========================================================================
print("=" * 74)
print("  ABNAHME 0 — die vier Tangenten gegen den Differenzenquotienten")
print("-" * 74)
print("  zentraler Differenzenquotient, Schrittweite h = 1e-6 V;")
print("  verglichen wird RELATIV, also |analytisch/numerisch - 1|.")

groesste_ableitung = 0.0
h = 1e-6
for kartenname, M in MODELLE_BJT.items():
    schlimmste = 0.0
    for v1 in np.arange(0.40, 0.86, 0.02):        # 23 Basisspannungen
        for v2 in (0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0):
            g = M.tangenten(v1, v2)
            num = []
            for k in (0, 1):
                a = [v1, v2]
                b = [v1, v2]
                a[k] += h
                b[k] -= h
                Ia = M.stroeme(*a)
                Ib = M.stroeme(*b)
                num.append(((Ia[0] - Ib[0]) / (2 * h),
                            (Ia[1] - Ib[1]) / (2 * h)))
            # num[k] = (dI_C/dv_k, dI_B/dv_k)
            ana = ((g[0], g[2]), (g[1], g[3]))    # ana[k] = (dI_C, dI_B)
            for k in (0, 1):
                for s in (0, 1):
                    n_, a_ = num[k][s], ana[k][s]
                    if abs(n_) < 1e-13:
                        continue
                    schlimmste = max(schlimmste, abs(a_ / n_ - 1.0))
    print(f"    {kartenname:15s}: groesste relative Abweichung "
          f"{schlimmste:.2e}")
    groesste_ableitung = max(groesste_ableitung, schlimmste)
pruefe("alle vier Ableitungen, alle vier Karten",
       groesste_ableitung, 1e-5)

# =========================================================================
print()
print("=" * 74)
print("  ABNAHME 1 — Arbeitspunkt der Emitterschaltung gegen kap08_rechnung.py")
print("-" * 74)

sim = Simulator(netz("bjt_fixedbias.netz"), dt=1e-3)
sim.schritt()
m = sim.messwerte("T1")
it1 = sim.it_groesste

# Seine Zahlen, aus kap08_rechnung.py mit voller Stellenzahl ausgegeben.
SEIN = {"u_be": 0.752165524359, "u_be_eff": 0.742407946744,
        "u_ce": 12.491100560417, "i_b": 6.505051743007e-04,
        "i_c": 1.250889943958e-01, "beta_eff": 181.122253750}
RB_KAP08 = 37275.390625

print(f"  Netzliste bjt_fixedbias.netz, R_B = {RB_KAP08} Ohm, Modell BC547")
print(f"  Newton: {it1} Durchgaenge vom kalten Start u = 0 "
      f"(die Daempfung d_max = 0,5 V braucht sie, um 25 V hochzulaufen),")
print(f"          Abbrueche an der Grenze max_it = {sim.max_it}: "
      f"{sim.abbrueche}")
print()
print(f"  {'Groesse':12s} {'Simulator':>20s} {'kap08_rechnung.py':>20s} "
      f"{'Stellen':>9s}")
schlechteste_stellen = 99.0
for k, einheit in (("u_be", "V"), ("u_be_eff", "V"), ("u_ce", "V"),
                   ("i_b", "A"), ("i_c", "A"), ("beta_eff", "-")):
    ist, soll = m[k], SEIN[k]
    rel = abs(ist - soll) / abs(soll)
    stellen = -np.log10(rel) if rel > 0 else 16.0
    schlechteste_stellen = min(schlechteste_stellen, stellen)
    print(f"  {k:12s} {ist:20.12g} {soll:20.12g} {stellen:9.1f}")
print()
print("  Uebereinstimmung mit seinem Programm: mindestens "
      f"{schlechteste_stellen:.1f} gueltige Stellen in jeder Groesse.")
pruefe("gueltige Stellen gegen kap08_rechnung.py (negativ gezaehlt)",
       -schlechteste_stellen, -5.0, "Stellen")

# --- und jetzt scharf: seine Gleichungen MIT GMIN ------------------------
# Der Simulator setzt 1e-9 S von jedem KNOTEN nach Masse. Die Knoten b
# und c tragen also je einen zusaetzlichen Ableitstrom GMIN*u, der in
# seinen Maschengleichungen fehlt. Der INNERE Basisknoten B' bekommt kein
# GMIN (er liegt hinter den Knotenindizes), V_BE,eff = V_BE - I_B*R_BI
# gilt also unveraendert.
M547 = MODELLE_BJT["BC547"]
VCC = VBB = 25.0
RC = 100.0


def F_hand(x):
    """Seine beiden Residuen (Kap. 8.4), ergaenzt um die GMIN-Ableitung."""
    vbe, vce = x
    ib = (VBB - vbe) / RB_KAP08 - GMIN * vbe      # Knotenbilanz b
    ic = (VCC - vce) / RC - GMIN * vce            # Knotenbilanz c
    veff = vbe - ib * M547.RBI
    IC, IB = M547.stroeme(veff, vce)
    return np.array([ib - IB, ic - IC])


def J_hand(x):
    """Seine Jacobi-Matrix (Kap. 8.6), um die GMIN-Glieder ergaenzt.
    Sie bestimmt nur, WIE SCHNELL Newton ankommt, nicht WO — die Loesung
    haengt allein an F = 0 (sein Satz in Kapitel 8.6)."""
    vbe, vce = x
    ib = (VBB - vbe) / RB_KAP08 - GMIN * vbe
    veff = vbe - ib * M547.RBI
    g11, g12, g21, g22 = M547.tangenten(veff, vce)
    dveff = 1.0 + M547.RBI * (1.0 / RB_KAP08 + GMIN)
    return np.array([
        [-1.0 / RB_KAP08 - GMIN - g21 * dveff, -g22],
        [-g11 * dveff, -1.0 / RC - GMIN - g12]])


# Startwert: SEINE Loesung. Gesucht ist ja nicht, ob Newton von weit weg
# herfindet (das zeigt Abnahme 1 am Simulator), sondern wo die um GMIN
# ergaenzten Gleichungen ihre Nullstelle haben.
x = np.array([SEIN["u_be"], SEIN["u_ce"]])
for _ in range(100):
    r = F_hand(x)
    if np.linalg.norm(r) < 1e-16:
        break
    x = x + np.linalg.solve(J_hand(x), -r)
vbe_h, vce_h = x
ib_h = (VBB - vbe_h) / RB_KAP08 - GMIN * vbe_h
ic_h = (VCC - vce_h) / RC - GMIN * vce_h
print()
print("  Dieselben Gleichungen von Hand, aber MIT GMIN gerechnet:")
print(f"    V_BE = {vbe_h:.12f} V   Simulator {m['u_be']:.12f} V")
print(f"    V_CE = {vce_h:.12f} V   Simulator {m['u_ce']:.12f} V")
print(f"    I_C  = {ic_h:.12e} A   Simulator {m['i_c']:.12e} A")
gr = max(abs(vbe_h - m["u_be"]), abs(vce_h - m["u_ce"]))
pruefe("gegen seine Gleichungen mit GMIN", gr, 1e-9, "V")
print("  Damit ist der Rest gegen sein Programm benannt und nicht nur")
print("  klein: es ist GMIN, nichts sonst.")

# =========================================================================
print()
print("=" * 74)
print("  ABNAHME 2 — Ausgangskennlinienfeld gegen die Messung Ic_Vce.txt")
print("-" * 74)
print("  BC337, Modellkarte aus bjt_fit.py (globaler Fit).")
print("  Ausgewertet wird der aktive Ast V_CE >= 1 V — genau die Grenze,")
print("  die sein bjt_fit.py auch zieht (der Saettigungsbereich gehoert")
print("  nicht in den Gueltigkeitsbereich seines Modells).")

mp = Messplatz("BC337")
tr, xs, ys = lade("Ic_Vce")
rel_aus = []
kurven = []
for lab, vce_m, ic_mA in zip(tr, xs, ys):
    ib_soll = trace_wert(lab) * 1e-6
    maske = np.isfinite(vce_m) & np.isfinite(ic_mA) & (vce_m >= 1.0)
    vv, im_, is_ = [], [], []
    for v, icm in zip(vce_m[maske], ic_mA[maske]):
        w = mp.bei_ib(ib_soll, float(v))
        vv.append(v)
        im_.append(icm * 1e-3)
        is_.append(w["i_c"])
        rel_aus.append((w["i_c"] - icm * 1e-3) / (icm * 1e-3))
    kurven.append((lab, np.array(vv), np.array(im_), np.array(is_)))
gr_aus, mi_aus = statistik("Ausgangskennlinienfeld", rel_aus, 6.0, 2.0)

# Die Abweichung ist SYSTEMATISCH (Mittelwert deutlich von Null weg), und
# das hat einen benennbaren Grund — also wird er nachgerechnet und nicht
# behauptet. Kapitel 6.8 schreibt I_C mit der KLEMMENspannung V_BE,
# Kapitel 8.4 und sein Programm mit V_BE,eff. Wir folgen Kapitel 8
# (siehe simulator.py, Abschnitt "DER TRANSISTOR"). Die BC337-Karte ist
# aber mit der Form aus 6.8 gefittet worden (bjt_fit.py, ic_model).
# Bei festem I_B unterscheiden sich beide Formen um genau den Faktor
# exp(I_B R_BI/(n V_T)) — das wird hier gemessen.
M337 = MODELLE_BJT["BC337"]
rel_68 = []
for (lab, vv, im_, is_), ib_lab in zip(kurven, tr):
    ib_soll = trace_wert(ib_lab) * 1e-6
    faktor = np.exp(ib_soll * M337.RBI / M337.nVT)
    rel_68 += list((is_ * faktor - im_) / im_)
rel_68 = np.array(rel_68)
print(f"    Mittelwert (mit Vorzeichen), Kapitel-8-Form : "
      f"{100*np.mean(rel_aus):+.2f} %")
print(f"    dieselben Punkte in der Kapitel-6.8-Form    : "
      f"{100*np.mean(rel_68):+.2f} %  "
      f"(mittlere |Abw.| {100*np.mean(np.abs(rel_68)):.2f} %)")
print("    Der systematische Anteil ist also genau der Formunterschied")
print("    exp(I_B R_BI/(n V_T)) zwischen Kapitel 6.8 und Kapitel 8.4,")
print("    und die Karte wurde mit der Form aus 6.8 gefittet. R_BI ist")
print("    aus diesen Daten ohnehin nicht bestimmbar (sein eigener")
print("    Identifizierbarkeitstest: Kostenfaktor 1,01).")
pruefe("Kapitel-6.8-Form trifft den Fit (Mittelwert)",
       abs(100 * np.mean(rel_68)), 0.5, "%")
print(f"    Newton im Sweep: {mp.sim.schritte} Arbeitspunkte, im Mittel "
      f"{mp.sim.it_summe/mp.sim.schritte:.2f} Durchgaenge, groesste "
      f"{mp.sim.it_groesste}, Abbrueche {mp.sim.abbrueche}")

# =========================================================================
print()
print("=" * 74)
print("  ABNAHME 3 — Eingangs- und Stromsteuerkennlinie")
print("-" * 74)

# --- 3a Eingangskennlinie (Gummel-Plot) gegen Ic_Vbe.txt -----------------
tr, xs, ys = lade("Ic_Vbe")
rel_ein = []
kurven_ein = []
for lab, vbe_m, ic_mA in zip(tr, xs, ys):
    vce = trace_wert(lab)
    if vce < 1.0:            # V_CE = 0 V ist nicht der aktive Bereich
        continue
    ic = ic_mA * 1e-3
    maske = (np.isfinite(vbe_m) & np.isfinite(ic)
             & (ic > 2e-5) & (ic < 2e-3))     # sein Auswertefenster
    vv, im_, is_ = [], [], []
    for v, i_m in zip(vbe_m[maske], ic[maske]):
        w = mp.punkt(float(v), vce)
        vv.append(v)
        im_.append(i_m)
        is_.append(w["i_c"])
        rel_ein.append((w["i_c"] - i_m) / i_m)
    kurven_ein.append((lab, np.array(vv), np.array(im_), np.array(is_)))
gr_ein, mi_ein = statistik("Eingangskennlinie (I_C ueber V_BE)",
                           rel_ein, 25.0, 10.0)
# In V_BE umgerechnet ist dieselbe Abweichung viel kleiner — die
# Exponentialfunktion uebersetzt Millivolt in Prozent (Rechenbeispiel 6.3).
nVT = MODELLE_BJT["BC337"].nVT
dv = nVT * np.log1p(np.array(rel_ein))
print(f"    dieselbe Abweichung als Spannung: groesste "
      f"{1e3*np.max(np.abs(dv)):.2f} mV, mittlere "
      f"{1e3*np.mean(np.abs(dv)):.2f} mV")
pruefe("Eingangskennlinie in V_BE gerechnet", 1e3 * np.max(np.abs(dv)),
       6.0, "mV")

# --- 3b Stromsteuerkennlinie gegen Ic_Ib.txt -----------------------------
tr, xs, ys = lade("Ic_Ib")
rel_str = []
kurven_str = []
for lab, ib_uA, ic_mA in zip(tr, xs, ys):
    vce = trace_wert(lab)
    maske = np.isfinite(ib_uA) & np.isfinite(ic_mA)
    bb, im_, is_ = [], [], []
    for ibm, icm in zip(ib_uA[maske], ic_mA[maske]):
        w = mp.bei_ib(float(ibm) * 1e-6, vce)
        bb.append(ibm)
        im_.append(icm * 1e-3)
        is_.append(w["i_c"])
        rel_str.append((w["i_c"] - icm * 1e-3) / (icm * 1e-3))
    kurven_str.append((lab, np.array(bb), np.array(im_), np.array(is_)))
gr_str, mi_str = statistik("Stromsteuerkennlinie (I_C ueber I_B)",
                           rel_str, 6.0, 2.0)

# =========================================================================
print()
print("=" * 74)
print("  ABNAHME 4 — Zeitverlauf: Emitterschaltung mit Koppelkondensator")
print("-" * 74)

DT = 1e-7                 # 5000 Schritte je Periode bei 2 kHz
F_SIG = 2000.0
PERIODEN = 12
N_SCHRITTE = int(round(PERIODEN / F_SIG / DT))

# --- 4a: Quelle aus — bleibt der Arbeitspunkt stehen? --------------------
# Die Netzliste startet den Koppelkondensator auf SEINEM V_BE aus
# kap08_rechnung.py. Der Simulator selbst steht mit GMIN um 34 nV daneben
# (Abnahme 1). Der Lauf muss also genau diese 34 nV abbauen und dann auf
# dem Arbeitspunkt der Abnahme 1 liegen bleiben — nicht auf irgendeinem.
AP_GLEICHSTROM = m["u_ce"]          # V_CE aus Abnahme 1, derselbe Loeser
sim = Simulator(netz("bjt_emitter_zeit.netz"), dt=DT)
sim.bauteil("Vs").U = 0.0
sim.schritt()
m0 = sim.messwerte("T1")
uc_start, ick0 = m0["u_ce"], sim.bauteil("Ck").uC
wandern = 0.0
for k in range(N_SCHRITTE):
    sim.schritt()
    wandern = max(wandern, abs(sim.messwerte("T1")["u_ce"] - uc_start))
uc0 = sim.messwerte("T1")["u_ce"]
d_anfang = abs(ick0 + SEIN["u_be"])
print(f"  Quelle abgeschaltet, {sim.schritte} Schritte a {DT*1e9:.0f} ns "
      f"({1e3*sim.t:.2f} ms):")
print(f"    u_Ck  Start {ick0:.12f} V  ->  Ende "
      f"{sim.bauteil('Ck').uC:.12f} V")
print(f"    V_CE  Start {uc_start:.9f} V  ->  Ende {uc0:.9f} V")
print(f"    Arbeitspunkt der Abnahme 1 (Gleichstrom): "
      f"{AP_GLEICHSTROM:.9f} V")
print(f"    groesste Wanderung von V_CE: {wandern:.3e} V")
print(f"    Erklaerung: der Anfangswert von u_Ck ist SEIN V_BE; der")
print(f"    Simulator steht mit GMIN {1e9*abs(ick0 - (-m['u_be'])):.1f} nV")
print(f"    daneben, und die Stufe verstaerkt das auf "
      f"{wandern/abs(ick0 + m['u_be']):.0f}-fach.")
pruefe("Endpunkt gleich dem Gleichstrom-Arbeitspunkt",
       abs(uc0 - AP_GLEICHSTROM), 1e-9, "V")
pruefe("Wanderung bleibt im Rahmen des Anfangswertfehlers",
       wandern, 1e-4, "V")

# --- 4b/4c: mit Signal ---------------------------------------------------
sim = Simulator(netz("bjt_emitter_zeit.netz"), dt=DT)
n = N_SCHRITTE
tt = np.zeros(n)
uce = np.zeros(n)
ubeeff = np.zeros(n)
ube = np.zeros(n)
ic_t = np.zeros(n)
ib_t = np.zeros(n)
uck = np.zeros(n)
for k in range(n):
    tt[k] = sim.t
    sim.schritt()
    w = sim.messwerte("T1")
    uce[k], ubeeff[k], ube[k] = w["u_ce"], w["u_be_eff"], w["u_be"]
    ic_t[k], ib_t[k] = w["i_c"], w["i_b"]
    uck[k] = sim.bauteil("Ck").uC

# (b) erfuellt die Loesung in JEDEM Schritt Lastgerade und Modell?
#     Die Modellgleichungen hier noch einmal getrennt hingeschrieben —
#     nicht der Stempel, sondern die nackte Formel aus Kapitel 6/8.
IS, NF, BF = M547.IS, M547.NF, M547.BF
VAF, VAR, IKF, VT = M547.VAF, M547.VAR, M547.IKF, M547.VT
E = np.exp(ubeeff / (NF * VT))
ic_formel = IS * E * (1.0 + uce / VAF)
q = np.sqrt(1.0 + np.maximum(ic_formel, 0.0) / IKF)
ib_formel = IS * q / BF * E * (1.0 + uce / VAR)
ic_last = (25.0 - uce) / RC - GMIN * uce          # Lastgerade mit GMIN
print()
print(f"  Mit Signal (20 mV, 2 kHz), {n} Schritte, {PERIODEN} Perioden:")
print(f"    Modellformel gegen Stempel   : "
      f"{np.max(np.abs(ic_formel - ic_t)):.3e} A / "
      f"{np.max(np.abs(ib_formel - ib_t)):.3e} A")
print(f"    Lastgerade gegen I_C         : "
      f"{np.max(np.abs(ic_last - ic_t)):.3e} A")
pruefe("Modellformel gegen Stempel (I_C)",
       np.max(np.abs(ic_formel - ic_t)), 1e-15, "A")
pruefe("Modellformel gegen Stempel (I_B)",
       np.max(np.abs(ib_formel - ib_t)), 1e-15, "A")
pruefe("Lastgerade in jedem Zeitschritt",
       np.max(np.abs(ic_last - ic_t)), 1e-11, "A")

# (c) Grosssignalverzerrung, von Hand nachrechenbar.
#
# Der Kollektorstrom ist  I_C = I_S e^{v1/(n V_T)} (1 + V_CE/V_A).
# Sein groesster Wert faellt mit dem groessten v1 und dem KLEINSTEN V_CE
# zusammen (die Lastgerade dreht das Vorzeichen um), der kleinste
# umgekehrt. Damit ist das Verhaeltnis der beiden Extremwerte
#
#     I_C,max     e^{v1,max/(n V_T)}   1 + V_CE,min/V_A
#     ------- =  ------------------- * ----------------
#     I_C,min     e^{v1,min/(n V_T)}   1 + V_CE,max/V_A
#             =  exp(2 u_hut/(n V_T)) * Early-Korrektur
#
# — zwei getrennt gemessene Huebe (der innere Spannungshub und der
# Kollektorhub) ergeben zusammen die Verzerrung. Auf Papier nachrechenbar.
letzte = tt >= (PERIODEN - 1) / F_SIG             # letzte volle Periode
u_hut = 0.5 * (ubeeff[letzte].max() - ubeeff[letzte].min())
uce_max, uce_min = uce[letzte].max(), uce[letzte].min()
verh_gemessen = ic_t[letzte].max() / ic_t[letzte].min()
verh_exp = np.exp(2.0 * u_hut / (NF * VT))
early_korr = (1.0 + uce_min / VAF) / (1.0 + uce_max / VAF)
verh_formel = verh_exp * early_korr
mittel = uce[letzte].mean()
oben = uce_max - mittel                  # Hub um die MITTELLINIE herum:
unten = mittel - uce_min                 # das ist das Mass der Verzerrung
print()
print("  Grosssignalverzerrung (letzte Periode):")
print(f"    innerer Hub u_hut(V_BE,eff)     = {1e3*u_hut:.4f} mV")
print(f"    I_C,max / I_C,min  gerechnet    = {verh_gemessen:.9f}")
print(f"    exp(2 u_hut/(n V_T))            = {verh_exp:.9f}")
print(f"    mal Early-Korrektur {early_korr:.9f} = {verh_formel:.9f}")
unsym = 100 * (oben - unten) / (0.5 * (oben + unten))
print(f"    V_CE-Hub um die Mittellinie: oben {1e3*oben:8.2f} mV, unten "
      f"{1e3*unten:8.2f} mV")
print(f"    Unsymmetrie {unsym:+.2f} %  —  eine LINEARE Stufe haette hier")
print("    exakt 0,00 %. Diese Unsymmetrie IST die Grosssignalverzerrung,")
print("    und sie ist die der e-Funktion.")
pruefe("Verzerrung sichtbar (negativ gezaehlt)", -abs(unsym), -5.0, "%")
pruefe("nur exp(2 u_hut/(n V_T)) (ohne Early)",
       abs(verh_gemessen / verh_exp - 1.0), 1e-1)
pruefe("exp mal Early-Korrektur",
       abs(verh_gemessen / verh_formel - 1.0), 1e-5)

# Arbeitspunktverschiebung: die e-Funktion gleichrichtet, der Mittelwert
# des Kollektorstroms steigt. Das ist ein ECHTER Effekt, kein Fehler —
# also wird er gemessen und nicht behauptet. Geprueft wird, dass er
# EINGESCHWUNGEN ist: von der vorletzten zur letzten Periode darf sich
# der Mittelwert nicht mehr nennenswert aendern.
vorletzte = ((tt >= (PERIODEN - 2) / F_SIG)
             & (tt < (PERIODEN - 1) / F_SIG))
mittel_v = uce[vorletzte].mean()
ib_m, ube_m, v1_m = (ib_t[letzte].mean(), ube[letzte].mean(),
                     ubeeff[letzte].mean())
print()
print("  Der Arbeitspunkt unter Grosssignal (Selbstvorspannung):")
print(f"    Mittelwert von V_CE, letzte Periode   : {mittel:.6f} V")
print(f"                         vorletzte Periode: {mittel_v:.6f} V")
print(f"    Arbeitspunkt ohne Signal              : {uc0:.6f} V")
print(f"    Verschiebung {1e3*(mittel-uc0):+.1f} mV")
print(f"    mittleres V_BE {1e3*ube_m:.4f} mV gegen {1e3*m['u_be']:.4f} mV "
      f"ohne Signal  ({1e3*(ube_m-m['u_be']):+.3f} mV)")
print(f"    mittleres I_C  {1e3*ic_t[letzte].mean():.4f} mA gegen "
      f"{1e3*m['i_c']:.4f} mA  -> ueber R_C genau die "
      f"{1e3*RC*(m['i_c']-ic_t[letzte].mean()):+.1f} mV oben")
print("    Das ist KEIN Wegdriften, sondern die Selbstvorspannung des")
print("    Koppelkondensators: die e-Funktion hebt den MITTLEREN")
print("    Basisstrom, und weil der Kondensator keinen Gleichstrom")
print("    traegt, muss die Gleichspannung an der Basis so weit sinken,")
print("    bis R_B den geforderten Mittelwert wieder liefert.")

# Nachweis, dass der Koppelkondensator keinen Gleichstrom traegt: die
# Knotenbilanz an b, ueber eine volle Periode gemittelt. Was der
# Kondensator im Mittel doch noch fuehrt, steht exakt in der Aenderung
# seines Zustands: I_Ck,mittel = C * du_Ck / T.
T_per = 1.0 / F_SIG
idx = np.where(letzte)[0]
d_uck = uck[idx[-1]] - uck[idx[0] - 1]
ick_mittel = sim.bauteil("Ck").C * d_uck / T_per
RB_ext = sim.bauteil("RB").R
bilanz = (25.0 - ube_m) / RB_ext - GMIN * ube_m - ick_mittel
print()
print(f"    Knotenbilanz b im Periodenmittel:")
print(f"      I_B                          = {ib_m:.9e} A")
print(f"      (V_CC - V_BE)/R_B - GMIN V_BE - I_Ck = {bilanz:.9e} A")
print(f"      (der Kondensator fuehrt im Mittel noch {ick_mittel:.2e} A,")
print(f"       das ist genau C du_Ck/T aus seinem Zustandswert)")
pruefe("Koppelkondensator traegt keinen Gleichstrom",
       abs(ib_m - bilanz), 1e-9, "A")
pruefe("Arbeitspunkt eingeschwungen (Periode zu Periode)",
       abs(mittel - mittel_v), 2e-3, "V")

# =========================================================================
print()
print("=" * 74)
print("  ABNAHME 5 — Gegenprobe mit seinen LTspice-Dateien")
print("-" * 74)
lt = {}
abbruch_gesamt = 0
for datei, asc, karte in (("bjt_ltspice_1d.netz", "Eigen_RW_1c/_1d",
                           "Demo_einfach"),
                          ("bjt_ltspice_1e.netz", "Eigen_RW_1e/_1f",
                           "Demo_erweitert")):
    s = Simulator(netz(datei), dt=1e-3)
    s.schritt()
    w = s.messwerte("T1")
    rb = s.bauteil("R2").R
    lt[karte] = (rb, w)
    abbruch_gesamt += s.abbrueche
    print(f"  {asc:16s}  R2 = {rb:8.0f} Ohm, .model {karte}")
    print(f"      Simulator: V_CE = {w['u_ce']:.4f} V   "
          f"V_BE = {1e3*w['u_be']:.2f} mV   I_C = {1e3*w['i_c']:.3f} mA")
    print(f"      sein Ziel: V_CE = V_CC/2 = 12,5000 V   "
          f"Abweichung {w['u_ce']-12.5:+.4f} V "
          f"({100*(w['u_ce']-12.5)/12.5:+.2f} %)")
pruefe("Eigen_RW_1e (erweiterte .model-Karte) gegen V_CC/2",
       abs(lt["Demo_erweitert"][1]["u_ce"] - 12.5), 0.05, "V")
# Kein Lauf dieses Programms darf Newton an der Grenze max_it abbrechen —
# sonst waere jede Zahl darueber wertlos.
abbruch_gesamt += sim.abbrueche + mp.sim.abbrueche
pruefe("kein Newton-Abbruch an max_it in irgendeinem Lauf (negativ)",
       -1.0 if abbruch_gesamt == 0 else abbruch_gesamt, 0.0)
print()
print("  Die erweiterte Karte trifft seinen LTspice-Basiswiderstand; die")
print("  einfache nicht — der offene Punkt ist im Manuskript, Abschnitt")
print("  50.5, benannt und NICHT weggerechnet.")

# =========================================================================
#  Bilder
# =========================================================================
os.makedirs(BILDER, exist_ok=True)
plt.rcParams.update({"font.size": 11})

# --- 1  Ausgangskennlinienfeld, darunter die Abweichung ------------------
fig, ax = plt.subplots(2, 1, figsize=(8.6, 6.8), dpi=150, sharex=True,
                       gridspec_kw={"height_ratios": [3, 1]})
for lab, vv, im_, is_ in kurven:
    c = ax[0].plot(vv, 1e3 * im_, "o", ms=4, label=lab)[0].get_color()
    ax[0].plot(vv, 1e3 * is_, "-", lw=2, color=c)
    ax[1].plot(vv, 100 * (is_ - im_) / im_, "-o", ms=3, lw=1.2, color=c)
ax[0].set_ylabel(r"$I_C$ in mA")
ax[0].set_title("Ausgangskennlinienfeld BC337: Messung (Punkte) gegen\n"
                "Netzlisten-Simulator (Linien)")
ax[0].grid(alpha=0.3)
ax[0].legend(fontsize=9, ncol=2, title="Basisstromstufe")
ax[1].axhline(0, color="black", lw=0.8)
ax[1].set_xlabel(r"$V_{CE}$ in V")
ax[1].set_ylabel("Abweichung\nin %")
ax[1].set_title(f"groesste {gr_aus:.2f} %, mittlere {mi_aus:.2f} % "
                f"({len(rel_aus)} Punkte)", fontsize=10)
ax[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(BILDER, "BJT_Ausgangskennlinienfeld.png"))
plt.close(fig)

# --- 2  Eingangs- und Stromsteuerkennlinie -------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.8), dpi=150)
for lab, vv, im_, is_ in kurven_ein:
    c = ax[0].semilogy(vv, im_, "o", ms=5, label=lab)[0].get_color()
    ax[0].semilogy(vv, is_, "-", lw=2, color=c)
ax[0].set_xlabel(r"$V_{BE}$ in V")
ax[0].set_ylabel(r"$I_C$ in A")
ax[0].set_title("Eingangskennlinie (Gummel-Plot), BC337\n"
                f"groesste Abweichung {1e3*np.max(np.abs(dv)):.2f} mV in "
                r"$V_{BE}$" f"  ({gr_ein:.1f} % in $I_C$)")
ax[0].grid(alpha=0.3, which="both")
ax[0].legend(fontsize=9)
for lab, bb, im_, is_ in kurven_str:
    c = ax[1].plot(bb, 1e3 * im_, "o", ms=5, label=lab)[0].get_color()
    ax[1].plot(bb, 1e3 * is_, "-", lw=2, color=c)
ax[1].set_xlabel(r"$I_B$ in $\mu$A")
ax[1].set_ylabel(r"$I_C$ in mA")
ax[1].set_title(f"Stromsteuerkennlinie, BC337\ngroesste {gr_str:.1f} %, "
                f"mittlere {mi_str:.1f} %")
ax[1].grid(alpha=0.3)
ax[1].legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(BILDER, "BJT_Eingang_Steuerkennlinie.png"))
plt.close(fig)

# --- 3  Zeitverlauf: Uebersicht oben, eine Periode gross unten -----------
ms = tt * 1e3
zoom = tt >= (PERIODEN - 2) / F_SIG                # die letzten zwei Perioden
fig = plt.figure(figsize=(11.6, 7.4), dpi=150)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.35], hspace=0.42,
                      wspace=0.26)

a0 = fig.add_subplot(gs[0, :])
a0.plot(ms, uce, lw=0.9, color="tab:blue")
a0.axhline(uc0, color="crimson", ls="--", lw=1.4,
           label=f"Arbeitspunkt ohne Signal {uc0:.3f} V")
a0.axhline(mittel, color="tab:orange", ls="-.", lw=1.6,
           label=f"Mittellinie mit Signal {mittel:.3f} V "
                 f"({1e3*(mittel-uc0):+.0f} mV: Selbstvorspannung)")
a0.set_xlim(0, ms[-1])
a0.set_xlabel("t in ms")
a0.set_ylabel(r"$V_{CE}$ in V")
a0.set_title("Emitterschaltung mit Koppelkondensator, expliziter Euler — "
             f"{n} Schritte zu {DT*1e9:.0f} ns\n"
             "Uebersicht: der Arbeitspunkt laeuft nicht weg, er stellt sich "
             "auf einen neuen Mittelwert ein")
a0.grid(alpha=0.3)
a0.set_ylim(uce.min() - 4.2, uce.max() + 0.6)   # Platz fuer die Legende
a0.legend(fontsize=9, loc="lower center", ncol=2, framealpha=0.95)

a1 = fig.add_subplot(gs[1, 0])
a1.plot(ms[zoom], 1e3 * (ubeeff[zoom] - ubeeff[letzte].mean()), lw=2,
        color="tab:green")
a1.axhline(0, color="black", lw=0.8)
a1.set_xlabel("t in ms")
a1.set_ylabel(r"$V_{BE,\mathrm{eff}}$ - Mittel  in mV")
a1.set_title("Eingang: der innere Spannungshub\n"
             fr"$\hat u = {1e3*u_hut:.2f}$ mV", fontsize=11)
a1.grid(alpha=0.3)

a2 = fig.add_subplot(gs[1, 1])
a2.plot(ms[zoom], uce[zoom], lw=2, color="tab:blue")
a2.axhline(mittel, color="tab:orange", ls="-.", lw=1.6)
a2.axhline(uce_max, color="0.55", ls=":", lw=1.2)
a2.axhline(uce_min, color="0.55", ls=":", lw=1.2)
a2.annotate("", xy=(ms[zoom][0] + 0.02, uce_max),
            xytext=(ms[zoom][0] + 0.02, mittel),
            arrowprops=dict(arrowstyle="<->", color="crimson", lw=1.4))
a2.annotate("", xy=(ms[zoom][-1] - 0.02, uce_min),
            xytext=(ms[zoom][-1] - 0.02, mittel),
            arrowprops=dict(arrowstyle="<->", color="crimson", lw=1.4))
a2.text(ms[zoom][0] + 0.05, 0.5 * (mittel + uce_max),
        f"oben\n{1e3*oben:.0f} mV", color="crimson", fontsize=10,
        va="center")
a2.text(ms[zoom][-1] - 0.07, 0.5 * (mittel + uce_min),
        f"unten\n{1e3*unten:.0f} mV", color="crimson", fontsize=10,
        va="center", ha="right")
a2.set_xlabel("t in ms")
a2.set_ylabel(r"$V_{CE}$ in V")
a2.set_title(f"Ausgang: {unsym:+.1f} % unsymmetrisch\n"
             "eine lineare Stufe haette hier 0,0 %", fontsize=11)
a2.grid(alpha=0.3)
fig.savefig(os.path.join(BILDER, "BJT_Zeitverlauf.png"),
            bbox_inches="tight")
plt.close(fig)

# --- 4  Die Ableitungsprobe: analytisch gegen Differenzenquotient --------
fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=150)
v1ax = np.arange(0.40, 0.861, 0.01)
M = MODELLE_BJT["BC547"]
v2p = 12.5
namen = (r"$g_{11}=\partial I_C/\partial V_{BE,\mathrm{eff}}$",
         r"$g_{12}=\partial I_C/\partial V_{CE}$",
         r"$g_{21}=\partial I_B/\partial V_{BE,\mathrm{eff}}$",
         r"$g_{22}=\partial I_B/\partial V_{CE}$")
kurv = [[], [], [], []]
for v in v1ax:
    g = M.tangenten(v, v2p)
    nn = []
    for k in (0, 1):
        pl, mi2 = [v, v2p], [v, v2p]
        pl[k] += h
        mi2[k] -= h
        Ia, Ib_ = M.stroeme(*pl), M.stroeme(*mi2)
        nn.append(((Ia[0] - Ib_[0]) / (2 * h), (Ia[1] - Ib_[1]) / (2 * h)))
    fuer = ((g[0], nn[0][0]), (g[1], nn[1][0]),
            (g[2], nn[0][1]), (g[3], nn[1][1]))
    for i, (an, nu) in enumerate(fuer):
        kurv[i].append(abs(an / nu - 1.0) if nu != 0 else np.nan)
for i, nm in enumerate(namen):
    ax.semilogy(v1ax, kurv[i], "-o", ms=3, lw=1.5, label=nm)
ax.axhline(1e-5, color="crimson", ls="--", lw=1.4,
           label="Schranke der Abnahme 0")
ax.set_xlabel(r"$V_{BE,\mathrm{eff}}$ in V  (bei $V_{CE}=12{,}5$ V)")
ax.set_ylabel("relative Abweichung  |analytisch/numerisch - 1|")
ax.set_title("Abnahme 0: die vier von Hand hergeleiteten Ableitungen gegen\n"
             "den zentralen Differenzenquotienten (BC547, h = 1 uV)\n"
             f"ueber alle vier Parameterkarten groesste Abweichung "
             f"{groesste_ableitung:.1e}")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=9, loc="lower left", ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(BILDER, "BJT_Ableitungsprobe.png"))
plt.close(fig)

print()
print("  Vier Bilder in ../Bilder/ geschrieben:")
for b in ("BJT_Ausgangskennlinienfeld.png", "BJT_Eingang_Steuerkennlinie.png",
          "BJT_Zeitverlauf.png", "BJT_Ableitungsprobe.png"):
    print("    " + b)

print()
print("=" * 74)
if fehler:
    print(f"  ERGEBNIS: {fehler} Abnahme(n) DURCHGEFALLEN")
    raise SystemExit(1)
print("  ERGEBNIS: alle Abnahmen bestanden")
print("=" * 74)

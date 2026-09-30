# -*- coding: utf-8 -*-
"""
===========================================================================
 simulator.py — vom Beispiel zum Simulator: Netzliste + Stempelklassen
===========================================================================

 Verallgemeinerung des Rechenkerns der B4-Bruecke (bruecke_kern.py) auf
 beliebige Schaltungen aus fuenf Bauteilsorten:

     Q / V   Spannungsquelle (dc oder sinus)
     R       Widerstand
     D       Diode (Shockley, ohne Naeherung — Modelle aus bruecke_kern)
     C       Kondensator
     L       Induktivitaet
     T       npn-Transistor (Gummel-Poon) — spaeter danebengelegt, siehe
             den Abschnitt "DER TRANSISTOR" weiter unten und Teil XII
             des Manuskripts. Am Loesungsgang aendert er nichts.

 DIE METHODE IST UNVERAENDERT die des Manuskripts (Abschnitt 3 und 25):

   Alles steht in EINER Knotenleitwertmatrix  Y * u = i.
   Jedes Bauteil traegt dort nur seinen festen STEMPEL ein.

   1  KNOTENPOTENTIALVERFAHREN   stempelt die Matrix und loest — jeder
                                 Durchgang. Erweiterte Knotenanalyse:
                                 Quellen und Kondensatoren bekommen ihren
                                 Zweigstrom als eigene Unbekannte, damit
                                 auch ideale Quellen (Ri = 0) gehen.
   2  NEWTON-RAPHSON             bildet in jedem Durchgang die Dioden-
                                 tangenten neu (g_d, I_eq) und wiederholt
                                 Stempeln und Aufloesen, bis der
                                 Arbeitspunkt steht — innere Schleife.
   3  EULER                      schreibt NACH Newton die Zustandsgroessen
                                 fort — aeussere Schleife, einmal je
                                 Zeitschritt. Waehlbar EXPLIZIT (wie im
                                 Manuskript, Teil V) oder IMPLIZIT
                                 (Companion-Modell aus der Integrations-
                                 formel, siehe die Stempelklassen von
                                 Kondensator und Induktivitaet). Die
                                 Verschachtelung ist in beiden Faellen
                                 dieselbe; implizit ist bei steifen
                                 Schaltungen und Schwingkreisen stabil,
                                 wo explizit die Schrittweite begrenzt.

 Zustandsgroessen und ihre Stempel WAEHREND eines Zeitschritts:

   Kondensator  u_C bekannt und fest.  Zwangszeile  u_p - u_n = u_C
                mit dem Zweigstrom I_C als Zusatzunbekannter;
                danach Euler:  u_C <- u_C + dt * I_C / C.
   Spule        i_L bekannt und fest.  Eingepraegter Strom i_L in die
                beiden Knoten (rechte Seite);
                danach Euler:  i_L <- i_L + dt * (u_p - u_n) / L.

 Beides ist dieselbe Arbeitsteilung wie im Manuskript: Newton sieht einen
 festen Zustand, Euler sieht die fertig aufgeloesten Stroeme und
 Spannungen. Newton und Euler tauschen je Speicher genau zwei Zahlen.

 Aufgeloest wird mit dem Gauss-Verfahren mit Teilpivotierung
 (numpy.linalg.solve) — die Handaufloesung des Dashboards lohnt nur fuer
 die eine, fest verdrahtete 4x4-Matrix.

 NETZLISTE: eine Zeile je Bauteil — Name, Knoten, Wert. Knoten sind
 beliebige Namen, "0" ist der Bezugsknoten. Kommentare mit * oder #.
 Werte mit den ueblichen Vorsaetzen (p n u m k meg). Beispiel:

     * B4-Bruecke mit belastetem RC-Glied (die Schaltung des Manuskripts)
     Q1  q 0  sinus 30 50
     Ri  q a  1
     D1  a P  1N4148
     D2  b P  1N4148
     D3  0 a  1N4148
     D4  0 b  1N4148
     R1  P k  10
     C1  k 0  1000u
     RL  k 0  100
     Q1  haengt an q, Ri verbindet q mit a — zusammen ist das die Quelle
     mit Innenwiderstand zwischen a und b aus dem Manuskript.

 Die Abnahmen stehen in simulator_abnahme.py und werden dort nachgewiesen,
 nicht angenommen — darunter: die Bruecken-Netzliste liefert dasselbe
 Ergebnis wie bruecke_kern.py.

 Nur numpy.   Aufruf:  python simulator.py [netzliste]
===========================================================================
"""
import numpy as np

from bruecke_kern import MODELLE

GMIN = 1e-9          # Leitwert jedes Knotens gegen Masse [S]

_VORSATZ = {"meg": 1e6, "k": 1e3, "m": 1e-3, "u": 1e-6,
            "n": 1e-9, "p": 1e-12, "f": 1e-15}


def wert(text):
    """Zahl mit Vorsatz: '1000u' -> 1e-3, '2.2k' -> 2200, '10' -> 10."""
    t = text.strip().lower()
    for vs in sorted(_VORSATZ, key=len, reverse=True):
        if t.endswith(vs):
            return float(t[:-len(vs)]) * _VORSATZ[vs]
    return float(t)


# =====================  DIE STEMPELKLASSEN  ===============================
# Jede Klasse kennt nur ihre eigenen Eintraege in Y und i. Zeilenbedeutung:
# Summe der aus dem Knoten ABFLIESSENDEN Zweigstroeme = eingepraegter Strom.

class Widerstand:
    """Vier Eintraege mit G = 1/R."""

    def __init__(self, name, p, n, rest):
        self.name = name
        self.p, self.n = p, n
        self.R = wert(rest[0])
        if self.R <= 0.0:
            raise ValueError(f"{name}: Widerstand muss positiv sein")

    def stempeln(self, Y, i, u, t, dt):
        G = 1.0 / self.R
        p, n = self.p, self.n
        Y[p, p] += G; Y[n, n] += G
        Y[p, n] -= G; Y[n, p] -= G

    def residuum(self, F, u, t):
        I = (u[self.p] - u[self.n]) / self.R
        F[self.p] += I; F[self.n] -= I

    def messwerte(self, u):
        uR = u[self.p] - u[self.n]
        return {"u": uR, "i": uR / self.R}


class Spannungsquelle:
    """dc oder sinus; der Zweigstrom ist Zusatzunbekannte j (erweiterte
    Knotenanalyse), die Zwangszeile lautet u_p - u_n = u_q(t)."""

    def __init__(self, name, p, n, rest):
        self.name = name
        self.p, self.n = p, n
        art = rest[0].lower()
        if art == "dc":
            self.U, self.f = wert(rest[1]), 0.0
        elif art == "sinus":
            self.U, self.f = wert(rest[1]), wert(rest[2])
        else:                       # nackte Zahl = dc
            self.U, self.f = wert(rest[0]), 0.0
        self.j = None               # Index der Stromunbekannten

    def u_quelle(self, t):
        if self.f == 0.0:
            return self.U
        return self.U * np.sin(2.0 * np.pi * self.f * t)

    def stempeln(self, Y, i, u, t, dt):
        p, n, j = self.p, self.n, self.j
        Y[p, j] += 1.0; Y[n, j] -= 1.0      # Zweigstrom in den Knotensummen
        Y[j, p] += 1.0; Y[j, n] -= 1.0      # Zwangszeile u_p - u_n = u_q
        i[j] += self.u_quelle(t)

    def residuum(self, F, u, t):
        I = u[self.j]
        F[self.p] += I; F[self.n] -= I
        F[self.j] += u[self.p] - u[self.n] - self.u_quelle(t)

    def messwerte(self, u):
        return {"u": u[self.p] - u[self.n], "i": u[self.j]}


class Diode:
    """Shockley ohne Naeherung; je Newton-Durchgang die Tangente
    g_d = dI/du und I_eq = I(v) - g_d*v am Arbeitspunkt (Manuskript,
    Abschnitt 13). p = Anode, n = Kathode."""

    def __init__(self, name, p, n, rest):
        self.name = name
        self.p, self.n = p, n
        modell = " ".join(rest) if rest else "1N4148"
        if modell not in MODELLE:
            raise ValueError(f"{name}: unbekanntes Diodenmodell '{modell}' "
                             f"(bekannt: {', '.join(MODELLE)})")
        self.modell = MODELLE[modell]

    def stempeln(self, Y, i, u, t, dt):
        p, n = self.p, self.n
        v = u[p] - u[n]
        gd = self.modell.leitwert(v)
        Ieq = self.modell.strom(v) - gd * v
        Y[p, p] += gd; Y[n, n] += gd
        Y[p, n] -= gd; Y[n, p] -= gd
        i[p] -= Ieq; i[n] += Ieq

    def residuum(self, F, u, t):
        I = self.modell.strom(u[self.p] - u[self.n])
        F[self.p] += I; F[self.n] -= I

    def messwerte(self, u):
        v = u[self.p] - u[self.n]
        return {"u": v, "i": self.modell.strom(v)}


class Kondensator:
    """Zustandsgroesse u_C, Zweigstrom I_C als Zusatzunbekannte j.

    EXPLIZIT:  u_C waehrend des Zeitschritts fest — Zwangszeile
               u_p - u_n = u_C ; danach  u_C <- u_C + dt * I_C / C.
    IMPLIZIT:  Companion-Modell aus der Integrationsformel
               u_neu = u_alt + dt/C * I_neu  (Manuskript, Abschnitt 3):
               Leitwert g_C = C/dt parallel zur Stromquelle g_C*u_alt.
               Zweigzeile  I_C = g_C*(u_p - u_n) - g_C*u_C_alt ;
               danach  u_C <- u_p - u_n  (die fertig aufgeloeste neue
               Spannung — der implizite Euler-Schritt steckt im Stempel).
    """

    def __init__(self, name, p, n, rest):
        self.name = name
        self.p, self.n = p, n
        self.C = wert(rest[0])
        if self.C <= 0.0:
            raise ValueError(f"{name}: Kapazitaet muss positiv sein")
        self.uC = wert(rest[1]) if len(rest) > 1 else 0.0   # Anfangswert
        self.j = None

    def stempeln(self, Y, i, u, t, dt):
        p, n, j = self.p, self.n, self.j
        Y[p, j] += 1.0; Y[n, j] -= 1.0      # I_C in den Knotensummen
        if self.implizit:
            gC = self.C / dt
            Y[j, p] -= gC; Y[j, n] += gC    # I_C = gC*(u_p-u_n) - gC*u_alt
            Y[j, j] += 1.0
            i[j] -= gC * self.uC
        else:
            Y[j, p] += 1.0; Y[j, n] -= 1.0  # Zwangszeile u_p - u_n = u_C
            i[j] += self.uC

    def residuum(self, F, u, t):
        I = u[self.j]
        F[self.p] += I; F[self.n] -= I
        if self.implizit:
            gC = self.C / self._dt
            F[self.j] += I - gC * (u[self.p] - u[self.n]) + gC * self.uC
        else:
            F[self.j] += u[self.p] - u[self.n] - self.uC

    def euler(self, u, dt):
        if self.implizit:
            self.uC = u[self.p] - u[self.n]
        else:
            self.uC += dt * u[self.j] / self.C

    def messwerte(self, u):
        return {"u": self.uC, "i": u[self.j]}


class Induktivitaet:
    """Zustandsgroesse i_L (Strom von p nach n).

    EXPLIZIT:  i_L waehrend des Zeitschritts fest — eingepraegter Strom
               in beide Knoten; danach  i_L <- i_L + dt * (u_p - u_n)/L.
    IMPLIZIT:  Companion-Modell aus  i_neu = i_alt + dt/L * u_neu :
               Leitwert g_L = dt/L parallel zur Stromquelle i_alt;
               danach  i_L <- i_alt + g_L * (u_p - u_n).
    """

    def __init__(self, name, p, n, rest):
        self.name = name
        self.p, self.n = p, n
        self.L = wert(rest[0])
        if self.L <= 0.0:
            raise ValueError(f"{name}: Induktivitaet muss positiv sein")
        self.iL = wert(rest[1]) if len(rest) > 1 else 0.0   # Anfangswert

    def stempeln(self, Y, i, u, t, dt):
        p, n = self.p, self.n
        if self.implizit:
            gL = dt / self.L
            Y[p, p] += gL; Y[n, n] += gL
            Y[p, n] -= gL; Y[n, p] -= gL
        i[p] -= self.iL               # i_L (alt) fliesst aus p ab ...
        i[n] += self.iL               # ... und in n hinein

    def residuum(self, F, u, t):
        I = self.iL
        if self.implizit:
            I = I + self._dt / self.L * (u[self.p] - u[self.n])
        F[self.p] += I; F[self.n] -= I

    def euler(self, u, dt):
        self.iL += dt * (u[self.p] - u[self.n]) / self.L

    def messwerte(self, u):
        return {"u": u[self.p] - u[self.n], "i": self.iL}


# =====================  DER TRANSISTOR  ===================================
# Gummel-Poon nach der Darstellung von Prof. Wystup — Buchkapitel 6 (6.8)
# und, fuer die Schaltungsform, Kapitel 8 (8.4/8.6/8.7) mit dem Programm
# kap08_rechnung.py. Hergeleitet und abgenommen im Manuskript, Teil XII.
#
# Die Gleichungen, WOERTLICH aus seiner Darstellung (Kapitel 8.4):
#
#     V_BE,eff = V_BE - I_B * R_BI            (6.7, "innerer Basiswiderstand")
#     E        = exp(V_BE,eff / (n V_T))
#     I_C      = I_S        * E * (1 + V_CE/V_A )        (6.4, Early)
#     beta_eff = beta_F / sqrt(1 + I_C/I_KF)             (6.6, Webster)
#     I_B      = I_S/beta_eff * E * (1 + V_CE/V_AB)      (6.5, Basis-Early)
#
# ZWEI ANMERKUNGEN ZUR TREUE (beide im Manuskript, Abschnitt 47.3, belegt):
#
#  1. Kapitel 6.8 schreibt im KASTEN den Kollektorstrom mit der KLEMMEN-
#     spannung V_BE, Kapitel 8.4 und sein Programm kap08_rechnung.py
#     schreiben ihn mit V_BE,eff. Wir folgen Kapitel 8 — das ist die
#     Schaltungsform, an der die tragende Abnahme haengt, und es ist die
#     physikalisch zwingende: sitzt R_BI als echter Widerstand in der
#     Basiszuleitung, sieht AUCH der Transportstrom nur die innere
#     Spannung. Der Unterschied ist nicht klein: im Arbeitspunkt des
#     Kapitels 8 faellt an R_BI 9,8 mV ab, das sind 46 % in I_C.
#
#  2. R_BI wird NICHT per Fixpunkt aufgeloest, sondern als echter
#     Widerstand zwischen der Klemme B und einem INNEREN Basisknoten B'
#     gestempelt. Damit ist V_BE,eff = u(B') - u(E) eine gewoehnliche
#     Knotenspannung, die implizite Kopplung aus 6.7 verschwindet, und
#     Newton loest sie mit auf — genau das, was Kapitel 6.7 als den
#     sauberen Weg nennt ("numerisch durch einen Fixpunkt- oder
#     Newton-Loeser ... und genau so macht es SPICE intern").
#     Bei R_BI = 0 entfaellt der innere Knoten, B' = B.
#
# GUELTIGKEITSBEREICH (seiner, Kapitel 6.8): Vorwaerts-Aktivbetrieb,
# T = 300 K, Quasistatik (keine Sperrschichtkapazitaeten, keine
# Laufzeit), Niedriginjektion bis auf die ueber I_KF erfasste
# Hochstromkorrektur. KEIN Rueckwaertsanteil: das Bauteil kennt weder
# Saettigung noch Inversbetrieb. Was SPICE darueber hinaus kann und was
# hier fehlt, steht im Manuskript, Abschnitt 51.

EXP_MAX_BJT = 200.0      # Ueberlaufschutz im Exponenten, wie bei der Diode


class GummelPoon:
    """Ein Parametersatz (eine .model-Karte) und die Auswertung des
    Modells samt seiner vier Tangenten.

    Unabhaengige Groessen sind die beiden Spannungen

        v1 = V_BE,eff = u(B') - u(E)        (innere Basis-Emitter-Spannung)
        v2 = V_CE     = u(C)  - u(E)        (Klemmen-Kollektor-Emitter-Sp.)

    abhaengig sind die beiden Stroeme I_C (C -> E) und I_B (B' -> E).
    Die vier Ableitungen sind von Hand hergeleitet (Manuskript, 48) und
    werden in bjt_abnahme.py gegen den zentralen Differenzenquotienten
    geprueft.
    """

    def __init__(self, IS, NF, BF, VAF, VAR, IKF, RBI,
                 VT=0.025852, name="", quelle=""):
        self.IS, self.NF, self.BF = IS, NF, BF
        self.VAF, self.VAR, self.IKF = VAF, VAR, IKF
        self.RBI, self.VT = RBI, VT
        self.name, self.quelle = name, quelle

    @property
    def nVT(self):
        return self.NF * self.VT

    # ------------------------------------------------------------------
    def _kern(self, v1, v2):
        """Die gemeinsamen Zwischengroessen von Strom und Tangente."""
        e = v1 / self.nVT
        E = np.exp(e if e < EXP_MAX_BJT else EXP_MAX_BJT)
        fA = 1.0 + v2 / self.VAF              # Early-Faktor     (6.4)
        fB = 1.0 + v2 / self.VAR              # Basis-Early      (6.5)
        IC = self.IS * E * fA
        # Webster: beta_eff = beta_F / q ,  q = sqrt(1 + I_C/I_KF).
        # max(I_C, 0) wie in seinem Programm kap08_rechnung.py.
        q = np.sqrt(1.0 + max(IC, 0.0) / self.IKF)
        A = self.IS / self.BF * E
        IB = A * q * fB
        return E, fA, fB, IC, q, A, IB

    def stroeme(self, v1, v2):
        """(I_C, I_B) — die beiden Stroeme, ohne Naeherung."""
        _, _, _, IC, _, _, IB = self._kern(v1, v2)
        return IC, IB

    def tangenten(self, v1, v2):
        """Die vier partiellen Ableitungen, analytisch (Manuskript, 48):

            g11 = dI_C/dv1 = I_C / (n V_T)
            g12 = dI_C/dv2 = I_S E / V_A
            g21 = dI_B/dv1 = A fB (q/(n V_T) + dq/dv1)
            g22 = dI_B/dv2 = A   (q/V_AB     + fB dq/dv2)

        mit  dq/dv_k = (dI_C/dv_k) / (2 q I_KF)  fuer I_C > 0, sonst 0
        (der Knick kommt aus dem max(I_C,0) seines Programms).
        """
        E, fA, fB, IC, q, A, IB = self._kern(v1, v2)
        g11 = self.IS * E * fA / self.nVT          # = I_C/(n V_T)
        g12 = self.IS * E / self.VAF
        if IC > 0.0:
            dq1 = g11 / (2.0 * q * self.IKF)
            dq2 = g12 / (2.0 * q * self.IKF)
        else:
            dq1 = dq2 = 0.0
        g21 = A * fB * (q / self.nVT + dq1)
        g22 = A * (q / self.VAR + fB * dq2)
        return g11, g12, g21, g22

    def alles(self, v1, v2):
        """Stroeme UND Tangenten in einem Durchgang (der Stempel braucht
        beides und soll die Exponentialfunktion nur einmal rechnen)."""
        E, fA, fB, IC, q, A, IB = self._kern(v1, v2)
        g11 = self.IS * E * fA / self.nVT
        g12 = self.IS * E / self.VAF
        if IC > 0.0:
            dq1 = g11 / (2.0 * q * self.IKF)
            dq2 = g12 / (2.0 * q * self.IKF)
        else:
            dq1 = dq2 = 0.0
        g21 = A * fB * (q / self.nVT + dq1)
        g22 = A * (q / self.VAR + fB * dq2)
        return IC, IB, g11, g12, g21, g22

    def beta_eff(self, v1, v2):
        _, _, _, IC, q, _, _ = self._kern(v1, v2)
        return self.BF / q


UNENDLICH = 1e12          # "Parameter aus; Term faellt weg" (1 + v/1e12 = 1)

# Die Parameterkarten. JEDE Zahl stammt aus seinen Unterlagen oder aus
# der Ausgabe seiner Programme; die Herkunft steht dabei. Es wird hier
# NICHTS neu gefittet.
MODELLE_BJT = {
    # Buch, Kapitel 7.8, Tabelle "BC547 (eigener Kennlinienschreiber)";
    # identisch mit dem Parameterblock von kap08_rechnung.py, Zeile 20/21.
    "BC547": GummelPoon(
        IS=5.0e-14, NF=1.01, BF=290.0, VAF=95.0, VAR=190.0,
        IKF=0.08, RBI=15.0, VT=0.02586, name="BC547",
        quelle="Buch Kap. 7.8 (gemessen) / kap08_rechnung.py"),

    # Ausgabe des globalen Fits bjt_fit.py (Nelder-Mead ueber alle fuenf
    # gemessenen Kennlinienfelder), Lauf vom 29.09.2026. V_AB, R_BI und
    # I_KF sind laut seinem eigenen Identifizierbarkeitstest aus diesen
    # Daten NICHT bestimmbar (Kostenfaktor 1,00 / 1,01 / 1,00) — sie
    # stehen hier als das, was sie sind: gesetzte Werte.
    "BC337": GummelPoon(
        IS=4.765e-14, NF=1.004, BF=253.2, VAF=126.6, VAR=1.39e3,
        IKF=4.05, RBI=18.8, VT=0.025852, name="BC337",
        quelle="bjt_fit.py, globaler Fit (V_AB, R_BI, I_KF nicht bestimmbar)"),

    # Die beiden .model-Karten seiner LTspice-Dateien, woertlich.
    # Eigen_RW_1c.asc und _1d.asc:
    #   .MODEL simple_npn NPN(IS=1e-13 BF=200 NF=1 VAF=100)
    "Demo_einfach": GummelPoon(
        IS=1e-13, NF=1.0, BF=200.0, VAF=100.0, VAR=UNENDLICH,
        IKF=UNENDLICH, RBI=0.0, VT=0.025852, name="Demo_einfach",
        quelle="LTspice Eigen_RW_1c.asc / _1d.asc"),
    # Eigen_RW_1e.asc und _1f.asc:
    #   .MODEL simple_npn NPN(IS=1e-13 BF=200 NF=1 VAF=100 VAR=200 IKF=0.5 RB=10)
    "Demo_erweitert": GummelPoon(
        IS=1e-13, NF=1.0, BF=200.0, VAF=100.0, VAR=200.0,
        IKF=0.5, RBI=10.0, VT=0.025852, name="Demo_erweitert",
        quelle="LTspice Eigen_RW_1e.asc / _1f.asc"),
}


class Transistor:
    """npn-Transistor nach Gummel-Poon — der Stempel.

    Netzliste:   T1  <K> <B> <E>  <Modellname>
    Beispiel:    T1  c   b   0    BC547

    Genau nach dem Muster der Diode: je Newton-Durchgang werden am
    gerade gueltigen Arbeitspunkt die TANGENTEN gebildet und in die
    Knotenmatrix gestempelt, der Rest geht als ERSATZSTROMQUELLE auf die
    rechte Seite. Statt einer Tangente g_d sind es hier vier Leitwerte
    und statt einer Ersatzquelle zwei — mehr ist der Unterschied nicht.

    Fuer einen Strom I, der von Knoten p nach Knoten q fliesst und von
    der Spannung v = u(a) - u(b) abhaengt, lautet der Stempel

        Y[p,a] += g ; Y[p,b] -= g ; Y[q,a] -= g ; Y[q,b] += g

    und die Ersatzquelle  I_eq = I - sum_k g_k v_k  geht mit
    i[p] -= I_eq ; i[q] += I_eq  auf die rechte Seite. Fuer die Diode
    fallen p = a und q = b zusammen, und es bleiben die bekannten vier
    Kreuzungen. Hier sind es zwei Stroeme mal zwei Spannungen, also vier
    solcher Bloecke.
    """

    KNOTEN = 3                      # C, B, E — die Netzliste liest drei

    def __init__(self, name, c, b, e, rest):
        self.name = name
        self.c, self.b, self.e = c, b, e
        modell = " ".join(rest) if rest else "BC547"
        if modell not in MODELLE_BJT:
            raise ValueError(f"{name}: unbekanntes Transistormodell "
                             f"'{modell}' (bekannt: "
                             f"{', '.join(MODELLE_BJT)})")
        self.M = MODELLE_BJT[modell]
        self.modell = modell
        if self.M.RBI > 0.0:
            # innerer Basisknoten B' als Zusatzunbekannte (der Simulator
            # vergibt den Index, wie bei Quelle und Kondensator)
            self.j = None

    # ------------------------------------------------------------------
    @property
    def bi(self):
        """Index des inneren Basisknotens — ohne R_BI ist das die Klemme."""
        return self.j if getattr(self, "j", None) is not None else self.b

    def _spannungen(self, u):
        return u[self.bi] - u[self.e], u[self.c] - u[self.e]

    # ------------------------------------------------------------------
    def stempeln(self, Y, i, u, t, dt):
        c, e, bi = self.c, self.e, self.bi
        v1, v2 = self._spannungen(u)
        IC, IB, g11, g12, g21, g22 = self.M.alles(v1, v2)

        # I_C (c -> e) nach v1 = u(bi) - u(e)
        Y[c, bi] += g11; Y[c, e] -= g11
        Y[e, bi] -= g11; Y[e, e] += g11
        # I_C (c -> e) nach v2 = u(c) - u(e)
        Y[c, c] += g12; Y[c, e] -= g12
        Y[e, c] -= g12; Y[e, e] += g12
        # I_B (bi -> e) nach v1
        Y[bi, bi] += g21; Y[bi, e] -= g21
        Y[e, bi] -= g21; Y[e, e] += g21
        # I_B (bi -> e) nach v2
        Y[bi, c] += g22; Y[bi, e] -= g22
        Y[e, c] -= g22; Y[e, e] += g22

        # die beiden Ersatzstromquellen
        ICeq = IC - g11 * v1 - g12 * v2
        IBeq = IB - g21 * v1 - g22 * v2
        i[c] -= ICeq; i[e] += ICeq
        i[bi] -= IBeq; i[e] += IBeq

        # der Basisbahnwiderstand zwischen Klemme b und innerem Knoten B'
        if bi != self.b:
            G = 1.0 / self.M.RBI
            b = self.b
            Y[b, b] += G; Y[bi, bi] += G
            Y[b, bi] -= G; Y[bi, b] -= G

    def residuum(self, F, u, t):
        c, e, bi = self.c, self.e, self.bi
        v1, v2 = self._spannungen(u)
        IC, IB = self.M.stroeme(v1, v2)
        F[c] += IC; F[e] -= IC
        F[bi] += IB; F[e] -= IB
        if bi != self.b:
            IR = (u[self.b] - u[bi]) / self.M.RBI
            F[self.b] += IR; F[bi] -= IR

    def messwerte(self, u):
        v1, v2 = self._spannungen(u)
        IC, IB = self.M.stroeme(v1, v2)
        return {"u_be": u[self.b] - u[self.e],    # Klemmenspannung
                "u_be_eff": v1,                   # innere Spannung
                "u_ce": v2,
                "u_bc": u[self.b] - u[self.c],
                "i_c": IC, "i_b": IB, "i_e": IC + IB,
                "beta": (IC / IB if IB != 0.0 else float("nan")),
                "beta_eff": self.M.beta_eff(v1, v2),
                "u": v2, "i": IC}


_TYPEN = {"R": Widerstand, "Q": Spannungsquelle, "V": Spannungsquelle,
          "D": Diode, "C": Kondensator, "L": Induktivitaet,
          "T": Transistor}


# =====================  DER SIMULATOR  ====================================
class Simulator:
    """
    Netzliste einlesen, dann je Zeitschritt:

        1. Quellspannungen auswerten (steckt im Stempeln)
        2. Newton: Tangenten bilden, stempeln, Gauss-aufloesen —
           wiederholen, bis groesstes |F| < tol_F und |delta| < tol_u
        3. Euler: alle Zustandsgroessen (u_C, i_L) fortschreiben

    Die Unbekannten: erst die Knotenpotentiale (ohne Bezugsknoten "0"),
    dahinter die Zweigstroeme der Quellen und Kondensatoren. Der
    Bezugsknoten wird intern mitgefuehrt und seine Zeile/Spalte vor dem
    Aufloesen gestrichen.
    """

    def __init__(self, netzliste, dt=10e-6, verfahren="explizit",
                 tol_F=1e-12, tol_u=1e-9, max_it=200, d_max=0.5):
        # tol_u ist bewusst 1e-9 V (der Kern nimmt 1e-12): ein Knoten, der
        # wie q in der Bruecken-Netzliste zeitweise nur ueber GMIN = 1e-9 S
        # am Rest haengt, macht aus dem Rundungsrest der Strombilanz
        # (~1e-17 A) ein Potentialrauschen von ~1e-11 V, und die
        # delta-Pruefung kaeme nie unter 1e-12 (gemessen: Dreierzyklus um
        # 2,6e-11 V, 28,8 statt 3,9 Newton-Durchgaenge im Mittel). Die
        # Strombilanz selbst bleibt bei tol_F = 1e-12 A.
        if verfahren not in ("explizit", "implizit"):
            raise ValueError("verfahren: 'explizit' oder 'implizit'")
        self.verfahren = verfahren
        self.dt = dt
        self.tol_F, self.tol_u = tol_F, tol_u
        self.max_it, self.d_max = max_it, d_max

        # ---- Netzliste lesen ----
        self.bauteile = []
        self.knoten = {"0": 0, "K0": 0}         # Name -> Index; K0 = Masse
        naechster_knoten = 1
        for zeile in netzliste.splitlines():
            zeile = zeile.split("*")[0].split("#")[0].strip()
            if not zeile:
                continue
            teile = zeile.split()
            typ = _TYPEN.get(teile[0][0].upper()) if teile else None
            if typ is None:
                raise ValueError(f"unbekannter Bauteiltyp: '{teile[0]}' "
                                 f"(Anfangsbuchstabe R, Q/V, D, C, L oder T)")
            # Zahl der Anschluesse: zwei, wenn das Bauteil nichts anderes
            # sagt (alle urspruenglichen Klassen), drei beim Transistor.
            anz = getattr(typ, "KNOTEN", 2)
            if len(teile) < anz + 2:
                raise ValueError(f"Netzlistenzeile unvollstaendig: '{zeile}'")
            name = teile[0]
            anschluesse = teile[1:1 + anz]
            rest = teile[1 + anz:]
            for k in anschluesse:
                if k not in self.knoten:
                    self.knoten[k] = naechster_knoten
                    naechster_knoten += 1
            b = typ(name, *[self.knoten[k] for k in anschluesse], rest)
            b.implizit = (verfahren == "implizit")
            self.bauteile.append(b)

        # ---- Zusatzunbekannte (Zweigstroeme) vergeben ----
        self.n_knoten = naechster_knoten
        j = self.n_knoten
        for b in self.bauteile:
            if hasattr(b, "j"):
                b.j = j
                j += 1
        self.n_unbekannte = j

        self.speicher = [b for b in self.bauteile if hasattr(b, "euler")]
        for b in self.speicher:
            b._dt = dt                  # fuers Residuum der Companion-Zeile

        self.ruecksetzen()

    # ------------------------------------------------------------------
    def ruecksetzen(self):
        self.t = 0.0
        self.u = np.zeros(self.n_unbekannte)    # Newton-Start: Vorschritt
        self.schritte = 0
        self.it_summe = 0
        self.it_groesste = 0
        self.abbrueche = 0      # Newton an der Grenze max_it beendet

    # ------------------------------------------------------------------
    def _newton(self):
        """VERFAHREN 2 um VERFAHREN 1: stempeln, aufloesen, wiederholen.

        Geloest wird die POTENTIALFORM  Y * u_neu = i ; der Abbruch prueft
        das wahre Residuum F — je Knoten die Summe der abfliessenden
        Zweigstroeme, je Zwangszeile die Spannungsbilanz — und die
        gedaempfte Aenderung delta (Manuskript, 12.6). F wird DIREKT aus
        den Zweigstroemen gebildet, nicht als Y*u - i: dort loeschen sich
        grosse Tangententerme (g_d von leitenden Dioden mal Potential)
        beinahe aus, und das Residuum kaeme nicht unter etwa 1e-10 —
        gemessen an der Bruecke: 28,8 statt 3,9 Newton-Durchgaenge.
        Mathematisch sind beide gleich, denn am Arbeitspunkt ist der
        Tangentenstrom gleich dem wahren Diodenstrom.
        """
        u = self.u
        frei = slice(1, None)                   # ohne Bezugsknotenzeile
        for it in range(1, self.max_it + 1):
            Y = np.zeros((self.n_unbekannte, self.n_unbekannte))
            i = np.zeros(self.n_unbekannte)
            F = np.zeros(self.n_unbekannte)
            for k in range(1, self.n_knoten):   # GMIN an jedem Knoten
                Y[k, k] += GMIN
                F[k] += GMIN * u[k]
            for b in self.bauteile:
                b.stempeln(Y, i, u, self.t, self.dt)
                b.residuum(F, u, self.t)

            u_neu = np.linalg.solve(Y[frei, frei], i[frei])
            delta = np.clip(u_neu - u[frei], -self.d_max, self.d_max)
            u[frei] += delta

            if (np.max(np.abs(F[frei])) < self.tol_F
                    and np.max(np.abs(delta)) < self.tol_u):
                break
        return it

    # ------------------------------------------------------------------
    def schritt(self):
        """Ein Zeitschritt: Newton (Zustand fest), dann Euler."""
        it = self._newton()
        self.it_summe += it
        if it > self.it_groesste:
            self.it_groesste = it
        if it >= self.max_it:
            self.abbrueche += 1
        for b in self.speicher:                 # VERFAHREN 3
            b.euler(self.u, self.dt)
        self.t += self.dt
        self.schritte += 1
        return it

    def lauf(self, t_ende):
        """Bis t_ende rechnen; Rueckgabe: Zeitachse und je Knoten der
        Potentialverlauf (dict Name -> Feld)."""
        n = int(round(t_ende / self.dt))
        tt = np.zeros(n)
        uu = {name: np.zeros(n) for name in self.knoten
              if self.knoten[name] != 0}
        for k in range(n):
            tt[k] = self.t
            self.schritt()
            for name, idx in self.knoten.items():
                if idx != 0:
                    uu[name][k] = self.u[idx]
        return tt, uu

    # ------------------------------------------------------------------
    def potential(self, knotenname):
        return self.u[self.knoten[knotenname]]

    def bauteil(self, name):
        for b in self.bauteile:
            if b.name == name:
                return b
        raise KeyError(name)

    def messwerte(self, name):
        """Spannung und Strom eines Bauteils im aktuellen Zustand."""
        return self.bauteil(name).messwerte(self.u)


# =====================  AUFRUF VON DER KOMMANDOZEILE  =====================
if __name__ == "__main__":
    import os
    import sys

    pfad = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "bruecke.netz")
    with open(pfad) as f:
        text = f.read()

    sim = Simulator(text)
    tt, uu = sim.lauf(0.2)

    print("=" * 62)
    print(f"  Netzliste: {os.path.basename(pfad)}")
    print(f"  Knoten   : {', '.join(k for k in sim.knoten if sim.knoten[k] != 0)}"
          f"   Unbekannte: {sim.n_unbekannte - 1}")
    print(f"  Schritte : {sim.schritte}   (dt = {sim.dt*1e6:.0f} us)")
    print(f"  Newton   : im Mittel {sim.it_summe/sim.schritte:.2f}, "
          f"groesste {sim.it_groesste}")
    for name in sorted(uu):
        print(f"  u({name})  am Ende: {uu[name][-1]:.9f} V")
    print("=" * 62)

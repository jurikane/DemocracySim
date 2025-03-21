### Warum ist die Granularität so grob, bzw. die Komplexität so niedrig?

Die Granularität ist weit weg von der Realität.
Das ist Absicht und hat vor allem zwei Gründe. 
Zum einen ist die Arbeit noch sehr grundlegend, 
weil es in der Literatur zu simulativen Vergleichen von Wahlverfahren noch nicht viel gibt,
auf dem eine höere Komplexität oder Granularität aufgebaut werden kann. 
Zum zweiten schwindet mit einer höheren Komplexität/Granularität sehr schnell die Interpretierbarkeit 
(und möglicherweise auch Reproduzierbarkeit). 
In Simulationen können aber nicht selten schon anhand sehr einfacher Modelle unerwartete Effekte 
und Mechanismen auftauchen. 
Das ist auch hier die Hoffnung.


### Was ist der Hauptkonflikt, den die Simulation untersucht?
Der Hauptkonflikt ist die Teilnahme an der Wahl. 
Da die einzelne Agentin zunächst (auf kurze Sicht) i.d.R. nicht erwarten kann 
ihre assets durch die Teilnahme zu steigern (außer alle Agenten denken so 
und die Wahlbeteiligung ist entsprechend niedrig), 
hat sie einen Anreiz sich die Kosten zu sparen und auf die Teilnahme zu verzichten 
(nach dem Motto "meine Stimme macht eh keinen Unterschied, dann muss ich auch nicht abstimmen"). 
Auf lange Sicht würde sich eine dauerhafte Nichtteilnahme aber vermutlich negativ für die Agentin auswirken, 
da zu erwarten ist, dass sich die Umgebung entgegen ihrer Interessen entwickelt.


Ein weiterer Konflikt ist die Abstimmung der Agentin selbst, 
also ob sie vorrangig ihr Wissen für die tatsächliche Verteilung (der Farben) in die Abstimmung einbringt 
(und damit allen hilft), oder eher ihren Interessen (eigenen Präferenzen) nach abstimmt 
(um einerseits selbst einen höheren Anteil an der Belohnung zu bekommen 
und andererseits die Umgebung zu ihren Gunsten zu beeinflussen).
Ob sie also eher "egoistisch" oder eher "altruistisch" bzw. "Gemeinwohl-orientiert" abstimmt.


### Wonach wird optimiert?
Für die Partizipation gibt es hoffentlich kein leicht zu berechnendes Optimum, 
da eine Simulation sonst überflüssig wäre, also das müssen wir meinem Verständnis nach verhindern 
(in dem Fall müssten wir das Modell komplexer machen). 
Die Optimierungsfunktion für das Training der Agenten ist nicht ganz leicht zu lösende Aufgabe. 
Gut wäre, wenn es ausreichte, die eigene Belohnung zu maximieren, 
weil das i.d.R. die Standardannahme ist. 
Ob das ausreicht oder die Agenten Modelle dann zu simpel werden ist noch nicht klar. 
Auf jeden Fall dürfen die Agenten weder zu intelligent, noch zu simpel sein. 
Vor allem weder zu kurz, noch zu weitsichtig. 
Das dürfte aber nicht nur eine Frage der Optimierungsfunktion sein, 
sondern auch der genauen Ausgestaltung des Trainings und der Input-Variablen. 
Auf jeden Fall ist das Training sehr wahrscheinlich der heikelste Part.

### Wie funktioniert die Ausschüttung der Belohnung(en)?
1. Nähe des Konsenses an der "Realität": 
   Jede Agentin (nicht nur die Teilnehmenden) erhält eine Belohnung $b_1$, 
   welche von dem Ergebnis der Wahl abhängt. 
   Je näher das Ergebnis der Wahl (die durch die Wahl geschätzte Häufigkeitsreihenfolge der Feldfarben)
   an der tatsächlichen Häufigkeitsreihenfolge der Feldfarben ist,
   desto größer $b_1$.
2. Nähe des Ergebnisses zur (fixen) persönlichen Präferenz (Persönlichkeit): 
   Jede Agentin, bekommt eine Belohnung $b_2$ 
   (wahrscheinlich mit $0 ≤ b_2 ≤ b_1$ oder sogar $-b_1 ≤ b_2 ≤ b_1$), 
   je nachdem wie gut das Ergebnis mit ihrer fixen persönlichen Präferenz 
   (also ihrer "Persönlichkeit", nicht der von ihr abgegebenen Präferenz) übereinstimmt.

Dabei soll $b_1$ den Umstand abbilden, dass die Beteiligung an einer Wahl einen 
(zwar eigentlich in seiner Höhe sehr subjektiven, aber dennoch vorhandenen) Aufwand bedeutet.
Und dass das Ergebnis bzw. die Folgen des Wahlausganges für alle Personen gleichermaßen gültig sind,
egal ob diese an der Wahl teilgenommen haben oder nicht.

Durch $b_2$ soll die Tatsache abgebildet werden, dass die Agenten auch eigene Vorlieben oder Bedürfnisse haben,
dass also das Ergebnis für sie persönlich lebensqualitätsbeeinflussend sein kann.
Außerdem ermöglicht $b_2$ die konfliktive Situation, 
dass die Wählenden eine Abwägung zwischen einer eher nach persönlicher Präferenz geprägten Stimmabgabe
und einer eher nach eigenem Wissen geprägten (tendenziell eher dem Gemeinwohl dienenden) Stimmabgabe treffen müssen.

### Welche Wahlverfahren werden untersucht?
Die Wahl (und Anzahl) der Wahlverfahren steht noch nicht ganz fest.
Im Moment ist geplant die folgenden Wahlverfahren zu untersuchen:
- "Plurality" als Standardverfahren 
- "Approval-Voting" da weitläufig als bestes Verfahren unter ComSoc-WissenschaftlerInnen angesehen
- "Kemeny" (Ebenfalls oft als bestes Verfahren angesehen, allerdings NP-Schwer). 

Und möglicherweise noch einige Standardverfahren. 
Interessant wären auch "exotischere" (weniger gut mathematisch untersuchte oder verbreitete) Verfahren 
wie "Systemisches-Konsensieren", "liquid-democracy" 
oder repräsentative Wahlverfahren (Wahl eines Gremiums) zu untersuchen. 

### Weitere bzw. weiter führende Forschungsfragen 
Ebenfalls interessant wäre am Ende der Vergleiche zu untersuchen, 
wie sich die Simulation verändert, wenn stets ein fixer Anteil an Agenten zufällig bestimmt wird, 
um (kostenlos oder sogar mit Aufwandsentschädigung) an der Wahr teilzunehmen 
(anstelle einer Freiwilligkeit welche mit Kosten verbunden ist).

Des Weiteren könnte untersucht werden was passiert, wenn Agenten zusätzliches "Wissen" (über Feldfarben) kaufen 
oder durch "laufen" bzw. springen "erkunden" können.

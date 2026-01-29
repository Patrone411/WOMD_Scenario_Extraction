# Betriebshandbuch
## OpenSCENARIO‑2.0‑Parser, Matching‑Pipeline und statistische Auswertung

Dieses Betriebshandbuch beschreibt die Nutzung, den Aufbau und das Zusammenspiel der im Rahmen dieser Arbeit entwickelten Softwarekomponenten. Der Fokus liegt auf der **reproduzierbaren Anwendung** der Pipeline sowie auf der **Nachvollziehbarkeit der Zuordnung zwischen Implementierung und den Kapiteln der Masterarbeit**. Die Darstellung folgt dabei bewusst einem sachlich‑technischen, akademischen Stil und verzichtet auf produktionsnahe Betriebsdetails.

---

## 1. Gesamtüberblick

Die Gesamtpipeline besteht aus drei logisch getrennten Hauptkomponenten:

1. **OSC‑2.0‑Parser** (Kapitel 4.2)
2. **Szenario‑Matching‑Pipeline** (Kapitel 4.6)
3. **Statistische Auswertung der gefundenen Szenarien** (Kapitel 5)

Die Komponenten sind lose gekoppelt. Insbesondere ist die statistische Auswertung vollständig von der Matching‑Logik entkoppelt und stellt kein funktionales Teilsystem der Szenarioerkennung dar.

---

## 2. OSC‑2.0‑Parser (Kapitel 4.2)

### 2.1 Zweck

Der OSC‑2.0‑Parser dient der strukturierten Analyse von OpenSCENARIO‑2.0‑Programmen und stellt eine erweiterte Version des Parsers aus dem Repository `carla‑simulator/scenario_runner` dar. Die Erweiterungen fokussieren sich auf:

- Sicherstellung von **Semantik und Typsicherheit**
- Ableitung expliziter **Block‑Pläne**
- Erzeugung einer **flachen Call‑Repräsentation** zur späteren Weiterverarbeitung

Das Handbuch beschreibt ausschließlich die **technische Umsetzung und Einordnung**; die formale Sprachdefinition und Motivation sind Gegenstand der Masterarbeit (Kapitel 4.2).

---

### 2.2 Einordnung und Abgrenzung

Der Parser baut auf dem Open‑Source‑Projekt

> **carla‑simulator/scenario_runner**

auf.

**Upstream‑Anteile:**
- Lexer‑ und Parser‑Infrastruktur
- AST‑Grundstruktur
- generische Visitor‑Mechanismen

**Beitrag dieser Arbeit:**
- explizite Semantik‑ und Typprüfungen für OSC 2.0
- Ableitung analysierbarer Zwischenrepräsentationen
- strukturierte Programmartefakte für nachgelagerte Verarbeitung

---

### 2.3 Systemüberblick

**Input:**
- OpenSCENARIO‑2.0‑Szenariodateien

**Zentrale Output‑Artefakte:**
- validierte Programmrepräsentation (`CompiledOSC`)
- Blockpläne
- flache Call‑Liste (`calls_flat`)
- abgeleitete Konfigurationsparameter (z. B. `min_lanes`)

---

### 2.4 Hauptkomponenten und Verarbeitungsschritte

#### 2.4.1 `OSCProgram`

**Modul:** `parser/program.py`

Zentrale Steuerkomponente des Parsers.

**Aufgaben:**
- Initialisierung des Parserlaufs
- Orchestrierung aller Verarbeitungsschritte
- Bereitstellung der finalen Artefakte

---

#### 2.4.2 AST‑Initialisierung: `ConfigInit (ASTVisitor)`

**Modul:** `parser/ast_visitors/config_init.py`

Initialer AST‑Visitor zur:
- Initialisierung von Symboltabellen
- Ableitung erster Typinformationen
- Vorbereitung der semantischen Analyse

---

#### 2.4.3 IR‑Ableitung: `IRLowering (ASTVisitor)`

**Modul:** `parser/ast_visitors/ir_lowering.py`

Transformiert den AST in eine vereinfachte **Intermediate Representation (IR)**.

---

#### 2.4.4 Semantik‑Registry

**Modul:** `parser/semantics/registry.py`

Zentrale Verwaltung aller Semantikregeln zur konsistenten Validierung.

---

#### 2.4.5 Semantik‑ und Typvalidierung

**Module:**
- `parser/semantics/semantic_validator.py`
- `parser/semantics/validate_from_ir.py`

Geprüft werden u. a.:
- Typkompatibilität von Akteuren, Aktionen und Modifiers
- Konsistenz relativer Akteurreferenzen
- semantische Vollständigkeit von Blöcken

---

#### 2.4.6 Ableitung von Blockplänen und flachen Calls

**Module:**
- `parser/ir/block_plan.py`
- `parser/ir/call_flattening.py`

Erzeugt explizite Analyseartefakte:
- Blockpläne
- flache Call‑Liste (`calls_flat`)

---

#### 2.4.7 Programmartefakt: `CompiledOSC`

**Modul:** `parser/compiled/compiled_osc.py`

Bündelt alle Ergebnisse eines Parserlaufs und stellt die **Schnittstelle zum Matcher** dar.

---

#### 2.4.8 Parameterableitung

**Modul:** `parser/ir/parameter_derivation.py`

Ableitung konfigurativer Parameter wie:

```python
min_lanes = get_min_lanes(scenarios_list, scenario_name="top", default=0)
```

Diese steuern nachgelagerte Matching‑Schritte.

---

### 2.5 Betrieb

Typischer Ablauf:
1. Laden einer OSC‑2.0‑Datei
2. Initialisierung von `OSCProgram`
3. AST‑Initialisierung und IR‑Lowering
4. Semantik‑ und Typvalidierung
5. Bereitstellung eines `CompiledOSC`‑Artefakts

Die resultierenden Datenstrukturen bilden die Grundlage für alle weiteren Matching‑Schritte.

---

## 3. Szenario‑Matching‑Pipeline (Kapitel 4.6)

### 3.1 Vorverarbeitung und Szenendaten

Die Vorverarbeitung der Waymo‑Open‑Motion‑Dataset‑Rohdaten erfolgt über Docker‑basierte Jobs. Sie stellt eine vorgelagerte Datenaufbereitung dar und ist **nicht Teil der eigentlichen Szenario‑Matching‑Pipeline**. Die Umwandlung der Rohdaten in segmentbasierte Szenenrepräsentationen wird durch das Modul

```
waymo_osc_extractor/scenario_processor2.py
```

realisiert und deckt die in den Kapiteln **4.3 bis 4.5.4** beschriebenen Schritte ab (Lane‑Graph‑Konstruktion, Segmentierung, Akteur‑Filterung, Persistierung).

Die Orchestrierung auf Kubernetes‑Ebene erfolgt über

```
waymo_osc_extractor/kube_runner.py
```

(Kapitel 4.5.4).

---

### 3.2 Hauptkomponenten des OSC Matchers

#### 3.2.1 Dockerfile

**Datei:** `Dockerfile`

Das Dockerfile definiert:

- die Laufzeitumgebung
- alle Abhängigkeiten für den Matcher
- den Einstiegspunkt zur Ausführung

Die Containerisierung stellt sicher, dass Matching‑Läufe reproduzierbar und unabhängig von der lokalen Umgebung sind.

---

#### 3.2.2 Ausführungslogik: `run_matching`

**Dateipfad:** `run_matching.py`

`run_matching` ist der zentrale Einstiegspunkt für den Matcher.

**Aufgaben:**

- Laden der Eingabedaten (OSC‑Programme, Szenen‑/Segmentdaten)
- Initialisierung der Matching‑Pipeline
- Iteration über **Szenen** (innerhalb der vorverarbeiteten Dateien) und deren Segmente
- Anwendung der Matching‑Kriterien
- Persistierung der Ergebnisse

Die in den Kapiteln **4.6.1 bis 4.6.6** der Masterarbeit beschriebenen Schritte werden vollständig durch `run_matching` orchestriert.

---

### 3.3 Matching‑Kernlogik (detailliert, Kapitel 4.6)

Dieser Abschnitt beschreibt die konkrete technische Umsetzung der Matching‑Logik und stellt die Verbindung zu den Unterkapiteln **4.6.1–4.6.6** der Masterarbeit her.

---

#### 3.3.1 Ableitung segmentweiser Feature‑Container (Kapitel 4.6.1)

**Modul:** `scenario_matching/features/adapters.py`

Die segmentweise Feature‑Extraktion erfolgt über die Klasse `TagFeatures`. Für jedes Straßensegment wird ein Feature‑Container erzeugt, der zeitdiskrete Merkmalsreihen für alle relevanten Akteure enthält.

**Enthaltene Merkmale (Auswahl):**

- Präsenz (`present`)
- Längs‑ und Querkoordinaten (`s`, `t`, `x`, `y`)
- Geschwindigkeit, Beschleunigung, Ruck
- Spurindizes und Relationen (`lane_idx`, `lat_rel`, `rel_position`)
- Interaktionsgrößen (z. B. TTC)

Diese Feature‑Container bilden die **einzige Datengrundlage** für das nachfolgende Matching.

---

#### 3.3.2 Rollenbildung und Domänenableitung (Kapitel 4.6.3)

**Zentrale Dateipfade:**

- `scenario_matching/matching/role_planning.py`
- `scenario_matching/matching/bindings.py`

Die Rollenbildung basiert auf den aus dem OSC‑Parser gelieferten Calls. Für jeden Call werden die beteiligten Rollen (z. B. Ego‑ und Referenzakteure) bestimmt.

**Ablauf pro Segment:**

1. Bestimmung der im Call verwendeten Rollen
2. Ableitung möglicher Akteurskandidaten pro Rolle (Domänen)
3. Vorfilterung nach Präsenz, Mindestdauer und optionalen Constraints

Die resultierenden Domänen definieren den Suchraum für die anschließende Bindungserzeugung.

---

#### 3.3.3 Call‑Auswertung pro Rollenbindung (Kapitel 4.6.3–4.6.4)

**Zentrale Dateipfade:**

- `scenario_matching/matching/collect_results.py`
- `scenario_matching/matching/match_single_call.py`
- `scenario_matching/matching/match_block.py`

Zentrale Funktion: `collect_results(...)`

Für jedes Segment und jeden Call wird iterativ über alle zulässigen Rollenbindungen iteriert:

- Rollenbindungen werden mit `bindings.py::enumerate_bindings(...)` erzeugt
- Optionale Überlappungsbedingungen zwischen Akteuren werden geprüft
- Für jede Rollenbindung wird ein `BlockQuery` ausgewertet

Die Auswertung wird über `match_for_binding(...)` angestoßen und liefert Trefferfenster (Start‑/Endframes) sowie optional Detailinformationen.

---

#### 3.3.4 Abbildung von Actions und Modifiers auf Feature‑Zeitreihen (Tabellen 4.1 & 4.2)

**Dateipfad:** `scenario_matching/matching/spec.py`

Die Datei `spec.py` definiert die formale Übersetzung von OpenSCENARIO‑2.0‑Elementen in zeitbasierte Prädikate.

**Zentrale Struktur:** `BlockQuery`

Unterstützt werden:

- aktionsspezifische Checks (z. B. `change_lane`, `assign_speed`, `keep_space_gap`)
- Modifier (z. B. `speed`, `position`, `lane`, `until`)
- Start‑, End‑ und Während‑Prüfungen (S/E/D‑Zerlegung)

Die Tabellen 4.1 und 4.2 der Masterarbeit entsprechen direkt den in `spec.py` implementierten Check‑Funktionen.

---

#### 3.3.5 Auswertung eines Action‑Calls und S/E/D‑Logik (Kapitel 4.6.4)

**Dateipfade:**

- `scenario_matching/matching/match_single_call.py`
- `scenario_matching/matching/match_block.py`
- `scenario_matching/matching/spec.py`

Ein Call entspricht einem **Action‑Call** aus OpenSCENARIO 2.0 und besteht aus genau einer Action sowie optional mehreren Modifiers.

**Verantwortlichkeiten:**

- `match_single_call.py::match_for_binding(...)` löst Rollen in konkrete Akteur‑IDs auf, kompiliert den Call zu einem `BlockQuery` und startet die S/E/D‑Auswertung.
- `match_block.py::match_block(...)` fungiert als Sliding‑Window‑Treiber für einen Action‑Call und erzeugt konsistente Trefferfenster.

**S/E/D‑Semantik innerhalb eines Action‑Calls:**

- **Start‑Checks (S):** Bedingungen am Startzeitpunkt `t0`
- **End‑Checks (E):** Bedingungen am Endzeitpunkt `t1`
- **During‑Checks (D):** Bedingungen über Frames innerhalb des Fensters
- **Window‑Checks:** Bedingungen über das gesamte Zeitfenster

---

#### 3.3.6 Kombination paralleler Action‑Calls zu Blocksignalen (Kapitel 4.6.5)

**Dateipfade:**

- `scenario_matching/matching/results/collect.py`
- `scenario_matching/matching/post/block_combine.py`
- `scenario_matching/matching/post/plan.py`

Mehrere Action‑Calls können als paralleler Block interpretiert werden.

**Technischer Ablauf:**

1. Atomare Auswertung pro Action‑Call → `PerCallSignal`
2. Kombination paralleler Calls mittels `block_combine.py::combine_parallel_block(...)`

Rollenbindungen werden dabei konsistent über Calls hinweg gehalten.

Das Ergebnis ist ein `BlockSignal`, das beschreibt, in welchen Zeitintervallen alle parallelen Call‑Bedingungen gemeinsam erfüllbar sind.

---

#### 3.3.7 Zeitfensterbildung, Aggregation und Persistierung (Kapitel 4.6.5–4.6.6)

Treffer einzelner Rollenbindungen werden zu Zeitintervallen aggregiert und vereinigt. Die Persistierung der Matchergebnisse erfolgt außerhalb der Matching‑Logik:

- `Dockerfile` (reproduzierbare Laufzeitumgebung)
- `run_matching.py` (Orchestrierung, Konfiguration und Ausgabe)

Die Matching‑Logik bleibt dadurch unabhängig von Infrastruktur‑ und Laufzeitdetails.

Ein Szenario gilt als erkannt, wenn **alle relevanten Action‑Calls eines Blocks bzw. Plans innerhalb eines konsistenten Zeitintervalls erfüllt sind**.

---

## 4. Definition eines Szenario‑Treffers (Kapitel 5)

Ein *Treffer* bezeichnet in dieser Arbeit kein einzelnes erfülltes Action‑Element, sondern ein **vollständig erkanntes Szenario**.

Jeder Treffer entspricht einem gültigen Zeitintervall innerhalb einer Szene, in dem ein vollständiges OpenSCENARIO‑2.0‑Szenario – d. h. die Kombination aller relevanten Action‑Calls eines Blocks bzw. Plans – gemäß der implementierten S/E/D‑Semantik sowie der Block‑Kombinationslogik erfüllt ist.

Einzelne Action‑Calls werden zunächst separat als Call‑Signale ausgewertet und anschließend zu Block‑ und Szenario‑Signalen zusammengeführt. **Erst diese aggregierten Signale werden als Treffer gezählt und ausgewertet.**

---

## 5. Statistische Auswertung der Szenario‑Treffer (Kapitel 5)

### 5.1 Datensammlung

Die Extraktion der Szenario‑Treffer erfolgt szenariospezifisch in den Modulen

```
scenario_matching/analysis_stats/stats_extractors_<SzenarioName>.py
```

Diese Skripte arbeiten ausschließlich auf den finalen Matching‑Ergebnissen (nach SED‑ und Block‑Kombination) und extrahieren unter anderem:

- Zeitintervalle (example windows)
- Szenen‑ und Segment‑IDs
- Rollenbindungen
- szenariospezifische Metriken

---

### 5.2 Histogramm‑basierte Analyse

Die Aggregation der extrahierten Zeitfenster über mehrere Szenen und TFRecords erfolgt in

```
stats_aggregation/<SzenarioName>/aggregate_example_windows_s3.py
```

Die erzeugten Histogramme bilden die Grundlage der quantitativen Analyse in Kapitel 5 und enthalten keine erneute Szenariointerpretation.

---

### 5.3 Typikalitätsanalyse

Während des Matchings wird pro TFRecord eine Statistikdatei (`stats_shard.json`) erzeugt. Die szenariospezifische Aggregation erfolgt über

```
stats_aggregation/<SzenarioName>/aggregate_stats_shard_s3.py
```

Die Typikalität eines Szenarios wird als relative Häufigkeit bezogen auf die betrachtete Datenbasis definiert.

---

## 6. Abgrenzung der Komponenten

- **Parser**: strukturelle und semantische Interpretation von OSC‑Programmen
- **Matching**: algorithmische Erkennung vollständiger Szenarien
- **Statistik**: rein nachgelagerte quantitative Auswertung

Es findet keine Vermischung dieser Verantwortlichkeiten statt.

---

## 7. Reproduzierbarkeit

Die vollständige Pipeline ist deterministisch und reproduzierbar, sofern identische Eingabedaten, Parser‑Versionen und Matching‑Konfigurationen verwendet werden.

---

*Ende des Betriebshandbuchs*


# Betriebshandbuch – waymo_osc_extractor

## 1. Zweck dieses Betriebshandbuchs

Dieses Dokument beschreibt den **Betrieb und die Verwendung** der im Rahmen der Masterarbeit entwickelten Softwarekomponenten zur Verarbeitung des Waymo Open Motion Dataset (WOMD).

Das Betriebshandbuch richtet sich explizit **nicht** an Entwickler zur Weiterentwicklung der Methodik, sondern an Leser (z. B. Prüfer), die nachvollziehen möchten:

* wie die Software ausgeführt wird,
* wie die Verarbeitung organisiert ist,
* und wie die Implementierung mit den Kapiteln der Masterarbeit zusammenhängt.

Die eigentliche wissenschaftliche Methodik wird in der Masterarbeit beschrieben; dieses Dokument dient ausschließlich der **technischen Einordnung und Nachvollziehbarkeit**.

---

## 2. Systemübersicht (Betriebssicht)

Die Pipeline dient der szenenweisen Verarbeitung von Waymo-TFRecord-Dateien und besteht aus folgenden Hauptschritten:

1. Laden von TFRecord-Rohdaten aus einem S3-Bucket
2. Iteration über alle enthaltenen Szenarien
3. Szenenweise Verarbeitung gemäß der in Kapitel 4 der Masterarbeit beschriebenen Methodik
4. Persistierung der Ergebnisse pro Szene

**Input:** Waymo TFRecord-Dateien (S3)
**Output:** Szenario-spezifische Ergebnisartefakte (`.pck`) im S3-Bucket

Die Ausführung erfolgt containerisiert (Docker) und kann optional über Kubernetes parallelisiert werden.

---

## 3. Betrieb und Verwendung

### 3.1 Docker

Das Docker-Image enthält alle notwendigen Abhängigkeiten zur Ausführung der Pipeline.

**Build:**

```bash
docker build -t waymo-osc-extractor:py38 .
```

**Start (lokal):**

```bash
docker run --rm \
  -e AWS_ACCESS_KEY_ID=... \
  -e AWS_SECRET_ACCESS_KEY=... \
  -e AWS_REGION=eu-west-1 \
  -e BUCKET=waymo \
  -e INPUT_PREFIX=tfrecords/training_tfexample.tfrecord \
  -e RESULT_PREFIX=results/test-run/ \
  -e SHARD_INDEX=0 \
  waymo-osc-extractor:py38 \
  python kube_runner.py
```

---

### 3.2 Kubernetes

Für skalierte Ausführung wird ein **Kubernetes Indexed Job** verwendet. Jeder Pod verarbeitet dabei **genau eine TFRecord-Datei**.

Die Auswahl der TFRecord-Datei erfolgt deterministisch über den von Kubernetes gesetzten Index (`JOB_COMPLETION_INDEX`). Optional kann ein globaler Startoffset gesetzt werden, um abgebrochene Läufe fortzusetzen.

---

## 4. Konfiguration (Environment-Variablen)

| Variable               | Beschreibung                                 |
| ---------------------- | -------------------------------------------- |
| `BUCKET`               | S3-Bucket mit den TFRecord-Dateien           |
| `INPUT_PREFIX`         | Prefix zur Auflistung der TFRecords          |
| `RESULT_PREFIX`        | Zielpfad für Ergebnisartefakte               |
| `JOB_COMPLETION_INDEX` | Automatisch gesetzter Pod-Index (Kubernetes) |
| `SHARD_INDEX`          | Manueller Index (Fallback ohne Kubernetes)   |
| `START_OFFSET`         | Offset zur Wiederaufnahme eines Laufs        |

---

## 5. Laufzeitverhalten und Fehlerbehandlung

* Fehler in einzelnen Szenarien werden **abgefangen** und führen nicht zum Abbruch des gesamten Jobs.
* Szenarien ohne valide Kartendaten werden übersprungen.
* Pods mit einem Index außerhalb des gültigen TFRecord-Bereichs beenden sich erfolgreich ohne Verarbeitung.

Dieses Verhalten ermöglicht robuste Langläufe und einfache Wiederaufnahme nach Abbrüchen.

---

## 6. Kompass: Zuordnung Code ↔ Masterarbeit

Die folgende Tabelle stellt die **Traceability** zwischen den Kapiteln der Masterarbeit und deren Implementierung im Code her.

| Abschnitt der Masterarbeit             | Zweck                                 | Implementierung                                 | Ergebnis / Artefakt            |
| -------------------------------------- | ------------------------------------- | ----------------------------------------------- | ------------------------------ |
| 4.3.1 Konstruktion des Lane-Graphen    | Aufbau des kartengestützten Suchraums | `ScenarioProcessor2` (`scenario_processor2.py`) | Lane-Graph                     |
| Abbildung 4.4 Root-/Branch-Sequenzen   | Suchraumreduktion                     | `ScenarioProcessor2`                            | Root-Sequenzen                 |
| 4.3.3 Aggregierung der Straßensegmente | Bildung zusammenhängender Segmente    | `ScenarioProcessor2`                            | `road_segs`                    |
| Segmentvalidierung                     | Entfernen ungültiger Segmente         | `ScenarioProcessor2`                            | gefilterte Segmente            |
| 4.3.4 Konvertierung in OpenDRIVE       | Export der Straßensegmente            | `xodr_exporter.py`                              | XODR-Repräsentation            |
| 4.4.1 Akteur-Preprocessing             | Reduktion auf relevante Akteure       | `ScenarioProcessor2`                            | kondensierte Akteurdaten       |
| 4.4.2 Segmentbasierte Filterung        | Inter-Akteur-Analyse                  | `ScenarioProcessor2`                            | Segment- und Interaktionsdaten |
| 4.5.2 Segmentbasierte Datenstruktur    | Zusammenführung der Ergebnisse        | `ScenarioProcessor2`                            | `result_dict`                  |
| 4.5.4 Verarbeitung & Persistierung     | Skalierte Ausführung und Speicherung  | `kube_runner.py`                                | `.pck`-Dateien in S3           |

**Abgrenzung:**

Komponenten wie `kube_runner.py`, Docker und Kubernetes dienen ausschließlich der **skalierbaren, reproduzierbaren Ausführung** der Pipeline. Sie sind **nicht Bestandteil der wissenschaftlichen Methodik oder Evaluation**, sondern stellen die technische Infrastruktur zur Umsetzung der in Kapitel 4 beschriebenen Verfahren dar.

---

## 7. Nicht-Gegenstand der Masterarbeit

Folgende Aspekte sind bewusst **nicht Teil der wissenschaftlichen Bewertung**:

* Containerisierung (Docker)
* Cloud-Speicher (AWS S3)
* Kubernetes-Parallelisierung
* Fehler- und Restart-Mechanismen

Diese Komponenten dienen ausschließlich dem stabilen Betrieb und der Reproduzierbarkeit der Experimente.

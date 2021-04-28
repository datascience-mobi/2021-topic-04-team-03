#Presentation Plan

1. Relevanz des Themas -> Liza
   Fragestellung: Zellkerne segmentieren
   -> Anwendungen: zB Input für Cell tracking Algorithmen vorzubereiten
    image segmentation - Thresholding, region growing, machine learning...
       image clipping -> problem: Wie wählt man den optimalen treshold?
        Otsu -> löst genau dieses Problem -> erklären (Zweck) + was ist Otsu (grob nicht den genauen Algorithmus)
        Otsu: Histogram bimodale Verteilung -> einfach, anderes -> schwerer) 
   
2. Procedure -> Laura
   0) Übersichtsfolie
   1) Input -> cell nuclei img -> Datensatz beschreiben
   2) pre-processing -> diverse filters (test several filters in order to reduce noise/sharpen edges)
   3) Otsu Thresholding + image clipping -> output
   4) dice score -> for evaluation
   5) Resultate mit Gruppen 4.4 und 4.5 vergleichen
    
2.1. Input - dataset -> Laura <br>
&nbsp;&nbsp;&nbsp;2.1.1. Beschreibung der Zellen - was sind HeLa Zellen - womit Bilder erzeugt wurden <br>
&nbsp;&nbsp;&nbsp;2.1.2. Schwierigkeiten der Bilder/Probleme die auftreten können <br>
&nbsp;&nbsp;&nbsp;2.1.3. Wofür Bilder aufgenommen wurden <br>
    - je eine Folie für jeden der drei Datensätze mit Beispielbild

2.2 preprocessing -> Liza <br>
&nbsp;&nbsp;&nbsp;2.2.1 Noise Reduction (Gaussian Filter/median Filter/?CLAHE) <br>
&nbsp;&nbsp;&nbsp;2.2.2 Higher Contrast (? further research) <br>
&nbsp;&nbsp;&nbsp;2.2.3 ? Edge Sharpening <br>

2.3 Otsu explanation (grobe Schritte) - image clipping -> Hannah
    - Separate Pixel in two classes: for all possible threshold values (below and above)
    - 'Between class variance' - formula
    - compute 'goodness' of thresholds, select maximum value - formula (anteil der between class varianz an Gesamtvarianz)
      0 <= goodness <= 1 (0: images with constant intensity, 1: binary images)
    - threshold value with highest 'goodness' is used for image clipping
    - image clipping: all intensities > threshold = 1, all intensities < threshold = 0 -> binary image as output - formula

2.4 Dice measure erklären -> Vero
    - Metric um verschiedene Segmentierungsalgorithmen zu vergleichen und zu bewerten
    - Vergleich von ground truth (manuell segmentierte Bilder) mit Resultaten
    - Schöne Bilder (Kreise) und schöne Formel (2*TP / (2*TP + FP + FN))
    - Ist eine von möglichen Methoden -> IoU, pixel acccuracy (in ein Satz erklären), Rand, Housdorff, NSD...

3. Timeline/Masterplan -> Liza <br>
   Milestones, wer macht was wann (D = derivable)
   - Was haben wir bis jz gemacht: Research Otsu, Dice, prepared the presentation <br>
   - get to know data: Histogramme (team) -> 19.05.
   - Otsu (Hannah, Liza) -> 26.05. -> D
   - Dice (Laura, Vero) -> 26.05. -> D
   - Filter -> 02.06. -> D <br>
   - Use our algorithm on images from BBBC -> 16.06.
   - other evaluation methods
     * research -> 09.06.
     * implementation -> 16.06.
       - IoU, pixel accuracy (team) -> D
       - Hausdorff (Hannah, Liza) -> D
       - NSD (Laura, Vero) -> D
   - compare results with group 4.4 and 4.5 -> 23.06
   - Jupyter Notebook -> 30.06.
   - Presentation -> 07.07.
    
? Nicolas fragen ob man Preprocessing nach den Entwicklung von Code für Otsu/Dice machen ok ist

4. Weitere Ideen -> Laura
    - 2D Otsu 
    - median Otsu
    - Algorithm for counting cells
    - Algorithm for drawing cell trajectory

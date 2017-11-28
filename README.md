# HappyDay

Ziel des Projektes ist es, anhand des Porträts einer Person den Gesichtsausdruck bzw. die Gemütslage zu identifizieren und ein passendes Emoji zuzuordnen. 
Dabei beschränken wir uns zunächst auf den Zustand lächeln. 

Ein typisches Szenario ist das Erstellen eines Fotos mit der Smartphone-Kamera in der im Projekt erstellten Android App. 
Daraufhin wird das Gesicht der fotografierten Person erkannt und zur Datenreduktion aus dem Bild ausgeschnitten. 
Der zugeschnittene Gesichtsausdruck wird zu einem Server gesendet. 
Auf diesem läuft ein Service, der aus dem Bildausschnitt aussagekräftige Gesichtszüge und Körperteile erkennt und diese definierten Emojis zuordnet. 
Realisiert wird die Gesichtserkennung mit einem CNN. 
Das Ergebnis der Auswertung wird an das Mobilgerät zurückgeschickt und dem User angezeigt. 
Dieser hat die Möglichkeit das empfangene Emoji zu bewerten und Feedback an den Server zu senden. 
Bei einer falschen Klassifizierung wird das Bild zum Training des CNN verwendet um dieses kontinuierlich zu verbessern.

# HappyDay Service

In diesem Repository befindet sich das Backend zu HappyDay. 

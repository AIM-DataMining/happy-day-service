# HappyDay Live

Ermöglicht die Live-Evaluierung auf dem PC mittels Webcam.
Durch die Applikation lassen sich verschiedene Netze trainieren
und das Ergebnis direkt mit einer Webcam evaluieren.


##Dependency
Als Laufzeitumgebung wird Anaconda in der 32 bit Variante eingesetzt eingesetzt.

Es gibt Abhängigkeiten zu folgenden Modulen die vor der Verwendung installiert werden müssen:

* numpy 1.13.3
* opencv 3.3.1
* tensorflow 1.4.0
* h5py 2.7.1
* Pillow 4.3.0

## Command Line Parameter:
Ausführen der Applikation mittels:

Training the data: train <filename> <CNN to use values between 1-3>
Example:
python Test train myModel 3
python Test train mySecondModel 2

Test the trained model: testloop <filename> <CNN to use values between 1-3>
Example:
python Test testloop myModel 3
python Test testloop mySecondModel 2


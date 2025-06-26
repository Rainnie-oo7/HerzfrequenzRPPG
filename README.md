Man kann Polygone verwenden. Ganz simple und einfache HR-Frequence Schätzung
1) Raw intensity data / 2) Bandpass Filter doesn't worked good / 3) FFT had I problem tha threshold-freqs, I define, could provide dramatically different Estimations (You have to know where "field" you have the HR/measure it gegen with a medical blood pressure cuff Device) / 4) I think it is too low-coded as other you couldn't use the Normalization, what's the original meaning by M.Avgs. (useless) / 5) Hamming Filter and Detrending delivered no good results neither.
Ein Machine Learning Ansatz wäre echt prima, es gibt viele Datensätze und Modelle z.B. von ubicomplab (edu.Washington) - I'll may look-forward i.t. Future

hr_estimation.py ist die ursprüngliche Prüfungsabgabe-Datei. Sie ist etwas falsch. Sie berücksichtigt aber nicht die tatsächliche Frameauslese, sondern eine Quarztaktzeitauslese und nicht die Rechnerleistung. (Mein PC las ein 30 fps Vid effektiv mit Verarbeitung, Plotting in 7 fps aus. 

hr_estimation_real_frame_time.py ist die eigentliche Datei. Sie speichert den Plot over Zeit gemäß ab und berücksichtigt die in die Zeitenreihe registrierten Zeiten (wahre Zeiten) des Videos mit den wahren Intensitätszeitwerten. Das Plotten dauert hierbei allerdings 30 fps / 7 fps etwa 5 mal länger, dafür aber genau.

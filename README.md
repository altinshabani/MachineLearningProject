# MachineLearningProject

Project for's Master's Course Machine Learning
Collaborators: Altin Shabani, Agnesa Sylejmani and Lulzim Shala. 

**Përshkrimi i Projektit: "Weather Prediction"**

Ky projekt është për parashikimin e ditës së ardhshme me shi duke përdorur teknikat e mësimit të makinës. Në këtë projekt, ne do të përdorim një dataset të Historisë së Motit për të ndihmuar në parashikimin e motit.
Modeli i mësimit të makinës që do të ndërtohet do të jetë në gjendje të analizojë dhe të përpunojë të dhënat për të identifikuar modelet dhe tendencat e motit në mënyrë që të parashikojë ditën e ardhshme me shi me një saktësi të lartë.
Përveç parashikimit të motit, ky projekt gjithashtu synon të rrisë ndërgjegjësinë për ndikimin e ndryshimeve klimatike dhe të ndihmojë njerëzit të marrin masa paraprake për të mbrojtur veten dhe pasurinë e tyre.

**Dataset**

Dataseti i të dhënave që përdoret në këtë projekt është seti i të dhënave të Historisë së Motit, i cili mund të shkarkohet nga Kaggle, linku i Historisë së Motit: https://www.kaggle.com/datasets/dataacademyhub/weatherhistory

**Përpunimi i të Dhënave**

Përpunimi i të dhënave është një hap thelbësor në këtë projekt.Kjo bëhet duke përdorur fajllin Python PjesaPare.py. Ky fajll fshin rreshtat me vlera që mungojnë për të siguruar saktësinë dhe integritetin e modelit të parashikimit si dhe largimin e të dhënave duplikate.

**Modelet ne Machine Learning**

Ne planifikojmë të përdorim  modele të ndryshme "Machine Learning" për të arritur qasjen tonë,Naive Bayes, Random Forest, K-Nearest Neighbors (KNN),Decision tree dhe Linear Regression. Këto modele do të tr![Uploading Temp-Hum.png…]()
ajnohen në bazën e të dhënave të parapërpunuara për të parashikuar motin ditët e ardhshme.

**Rezultatet**

Duke kontrolluar korrelacionin e variablave tona MaxTemp dhe MinTemp, mund të shohim se kur kemi temperatura të larta, është më e zakonshme që të mos ketë shi ditën e ardhshme.

![image](https://github.com/altinshabani/MachineLearningProject/assets/119792193/2ddeb3c8-860c-45bf-a921-130bc583224d)

-Naive Bayes-

![image](https://github.com/altinshabani/MachineLearningProject/assets/119792193/8b7c96ca-65ce-469c-aa16-29f8bb0c7801)

Accuracy:

![image](https://github.com/altinshabani/MachineLearningProject/assets/119792193/fcc472f3-832b-463c-b155-accb6bc791aa)

-Decission Tree-

![image](https://github.com/altinshabani/MachineLearningProject/assets/119792193/77a5bb71-5716-413b-bf83-c4b60563a3f8)


-KNN-

![image](https://github.com/altinshabani/MachineLearningProject/assets/119792193/6c65af3c-6115-4dcb-8614-0dc40ab7c710)

Accuracy:

![image](https://github.com/altinshabani/MachineLearningProject/assets/119792193/5a0b5dc1-a49c-4e4a-85c4-ebfff840cf80)



-Logistic Regression-

![image](https://github.com/altinshabani/MachineLearningProject/assets/119792193/b6f7d164-20c4-44fc-9d34-308c4e3f7046)

![image](https://github.com/altinshabani/MachineLearningProject/assets/119792193/b3b2ae1a-9ded-4352-8339-56305d73f482)


Looking at our Data Analysis, we can see that the temperature of the previous day is a important variable to get our variable target, but also PrecipType and Summary.






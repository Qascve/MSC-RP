This data file includes data on courtship rate, morphological characteristics and basal metabolic rate from a captive population of zebra finches. The data were collected by Wolfgang Forstmeier, Katrin Martin and Kimberley Mathot at the Max Planck Institute for Ornithology in Seewiesen, Germany.


File name: BMRdata.csv
Column headings and brief explanation

date		date of BMR measurements (dd/mm/yyyy)
Session8	ID of BMR measurement session, 8 sessions total (A through H).
Session2	ID of BMR measurements collapsed into 2 sessions (see text). A, B, D, E, F, G, H = 1, C = 2 
AnimalID	Unique individual ID
DOB		date of birth (dd/mm/yyyy)
age		age in days at BMR measurement
SEX		0 = female, 1 = male
F		Inbreeding coefficient from a 7-generation pedigree
Line		Selection line type (low, control or high courtship) and line number
MassEvening	body mass in grams taken in evening prior to the start of BMR measurements
time		Time of day at at evening body mass measurement in full days (where 0 = 00:00, 1 = 24:00)
MassMorning	body mass in grams taken in the morning immediately following the end of BMR measurements
MassatBMR	body mass at the time of lowest O2 consumption, assuming linear mass loss between evening and morning mass measurements
BMR		basal metabolic rate in mL O2 per min
	


File name: Pedigree.csv
Column headings and brief explanation

animal		Unique individual ID
mother		ID of mother
father		ID of father
Inbreeding	Inbreeding coefficient of the animal
sex		0 = female, 1 = male
DOB		date of birth (dd/mm/yyyy)



File name: Animalmodels.csv
Column headings and brief explanation

Inbreedingcoeff	Inbreeding coefficient of the animal
ageindays	age in days at measurement
timeofday	time of day at measurement in full days (where 0 = 00:00, 1 = 24:00)	
session		ID of measurement session, arbitrary coding within each phenotypic trait, same IDs for different traits do not refer to the same session
sex		0 = female, 1 = male	
testday		day of testing for courtship rate (order of tests within males within sessions)
animal		ID of the focal individual (to be linked to the pedigree)
mother		ID of the mother of the focal individual
PermEnv		ID of the focal individual (for the estimation of permanent environment effects)
BMR		basal metabolic rate in mL O2 per min
Mass		body mass in grams
Courtship	courtship rate (square-root of the number of seconds of courtship song within a 5-minute test)
tarsus		length of the tarsus in mm (including the joint with the tibiotarsus)
wing		flattened wing length in mm
BMRres		Residuals from a regression of BMR over Mass (with y = 0.03755x + 0.2257)
Line		ID of the selection line (0 refers to unselected individuals from the base population)


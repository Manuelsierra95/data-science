from functools import reduce

resources = {
        'age': 'age in years',
        'sex': {'male': 1,
            'female': 0},
        'cp': {'info': 'chest pain type',
			'typical angina':0,
			'atypical angina': 1,
			'non anginal pain': 2,
			'asymptomatic': 3},
        'trstbps': {'info': 'resting blood pressure',
			'is typical': '130-140'},
        'chol': {'info': 'serum cholestoral',
			'is bad': 200},
        'fbs': {'info': 'fasting blood sugar > 120',
			'True': 1, 'False': 0},
        'restecg': {'info': 'resting electrocardiographic',
			'nothing': 0,
			'ST-T abnormality': 1,
			'Possible or definite left ventricular hypertrophy': 2},
        'thalach': 'maximum heart rate achieved',
        'exang': {'info': 'exercise induced angina',
			'yes': 1,
			'no': 0},
        'oldpeak': 'ST depression induced by exercise relative to rest looks at stress of heart during excercise unhealthy heart will stress more',
		'slope': {'info': 'the slope of the peak exercise ST segment',
			'Upsloping: better heart rate with excercise (uncommon)': 0,
			'Flatsloping: minimal change (typical healthy heart)': 1,
			'Downslopins: signs of unhealthy heart': 2},
		'ca': 'number of major vessels (0-3) colored by flourosopy',
		'thal': {'info': 'thalium stress result',
			'normal': 13,
			'fixed defect': 6,
			'reversable defect': 7},
		'target': {'info': 'have disease or not',
			'yes': 1,
			'no': 0}
    }

def get_resource(d, *keys):
    # Extrae del dict la info segun la key de forma recursiva
    resources = []
    resources.append(keys)
    resources.append(reduce(lambda c, k: c.get(k, {}), keys, d))
    print(resources) # hacer el print mas bonito

# get_resource(resources, 'age')
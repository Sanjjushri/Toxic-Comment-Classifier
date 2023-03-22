'''
Created on 

Course work: 

@author: Sanjjushri

Source:
    
'''


import pickle


loaded_model = pickle.load(open('model_clf.pkl', 'rb'))
loaded_vectorizor = pickle.load(open('count_vect.pkl', 'rb'))


def predict_comment(comment):

    predicted_value = loaded_model.predict(loaded_vectorizor.transform([comment]))

    print(predicted_value)
    
    return predicted_value

def startpy():
    
    predict_comment("Wasn't me, asshole.  Check your fucking IP's before you go shooting off your mouth.")


startpy()
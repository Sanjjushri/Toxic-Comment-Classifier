'''
    
    Source:    
        
'''
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import os.path

CSV_FILE_PATH       = 'toxic-comment-dataset.csv'
MODEL_PICKLE_PATH   = 'mode_clf_5.pkl'
VECTOR_PICKLE_PATH  = 'count_vect_5.pkl'

class MLSingleton:

    __instance = None

    def __init__(self):
        """ Virtually private constructor. """
        if MLSingleton.__instance != None:
            raise Exception("This class is a singleton!")

        self.model_name         = MODEL_PICKLE_PATH
        self.vectorizer_name    = VECTOR_PICKLE_PATH

        try:
            # Load models from pickle
            self.model              = pickle.load(open(self.model_name, 'rb'))
            self.vectorizer         = pickle.load(open(self.vectorizer_name, 'rb'))
        except FileNotFoundError as file_err:
            print('Error while loading pickles : ', file_err)

        print('Loaded model and vectorizer')

        MLSingleton.__instance = self

    def train_model(self, force = False):

        if(not force):

            print(f'{MODEL_PICKLE_PATH} available? : ', os.path.isfile(MODEL_PICKLE_PATH))
            if(os.path.isfile(MODEL_PICKLE_PATH)):
                print('Skipped as the file is already there')
                return

        print('Training fresh model and dumping pickle')

        df = pd.read_csv(CSV_FILE_PATH)

        X_train, X_test, y_train, y_test = train_test_split(df['comment_text'], df['toxic'], random_state = 100, test_size = 0.1)

        count_vect          = CountVectorizer()
        X_train_counts      = count_vect.fit_transform(X_train)
        tfidf_transformer   = TfidfTransformer()
        X_train_tfidf       = tfidf_transformer.fit_transform(X_train_counts)

        local_model               = MultinomialNB()
        clf                 = local_model.fit(X_train_tfidf, y_train)
        # pred                = clf.predict(count_vect.transform(X_test))
        
        # print("Accuracy score: ", accuracy_score(y_test, pred))
        # print("  ")

        # print(classification_report(y_test, pred))
        # confusion_matrix(y_test, pred)

        # dump pickle
        pickle.dump(clf, open(MODEL_PICKLE_PATH, 'wb'))
        print('Dumped model into ', MODEL_PICKLE_PATH)

        # dump pickle count vectorizer
        pickle.dump(count_vect, open(VECTOR_PICKLE_PATH, 'wb'))
        print('Dumped vectorizer into ', VECTOR_PICKLE_PATH)

        # as fresh model created apply them
        self.model              = pickle.load(open(self.model_name, 'rb'))
        self.vectorizer         = pickle.load(open(self.vectorizer_name, 'rb'))

    @staticmethod 
    def getInstance():
        """ Static access method. """
        if MLSingleton.__instance == None:
            MLSingleton()

        return MLSingleton.__instance

    def classify_single_comment(self, comment):

        predicted_value = self.model.predict(self.vectorizer.transform([comment]))

        if predicted_value[0] == 1:
            # print('comment : ', comment)
            # print('toxic')
            return 'toxic'

        return 'general'

    def classify_comments(self, comment_json):

        comment_list = comment_json['comments']

        comment_new_list = []
        for c_dict in comment_list:
            for key, value in c_dict.items():
                print('key : ', key)
                print('val : ', value)
                # print(key, value)

                category = self.classify_single_comment(value)

                cnew_dict = {
                    "id" : key,
                    "comment" : value,
                    "cateogry" : category
                }

                comment_new_list.append(cnew_dict)

        return comment_new_list

def predict_and_dump():

    df = pd.read_csv(CSV_FILE_PATH)

    X_train, X_test, y_train, y_test = train_test_split(df['comment_text'], df['toxic'], random_state = 100, test_size = 0.1)

    count_vect          = CountVectorizer()
    X_train_counts      = count_vect.fit_transform(X_train)
    tfidf_transformer   = TfidfTransformer()
    X_train_tfidf       = tfidf_transformer.fit_transform(X_train_counts)

    model               = MultinomialNB()
    clf                 = model.fit(X_train_tfidf, y_train)
    pred                = clf.predict(count_vect.transform(X_test))
    
    # print("Accuracy score: ", accuracy_score(y_test, pred))
    # print("  ")

    # print(classification_report(y_test, pred))
    # confusion_matrix(y_test, pred)

    # dump pickle
    pickle.dump(clf, open(MODEL_PICKLE_PATH, 'wb'))
    print('Dumped model into ', MODEL_PICKLE_PATH)

    # dump pickle count vectorizer
    pickle.dump(count_vect, open(VECTOR_PICKLE_PATH, 'wb'))
    print('Dumped vectorizer into ', VECTOR_PICKLE_PATH)

def predict_comment(comment):

    loaded_model        = pickle.load(open(MODEL_PICKLE_PATH, 'rb'))
    loaded_vectorizor   = pickle.load(open(VECTOR_PICKLE_PATH, 'rb'))

    predicted_value     = loaded_model.predict(loaded_vectorizor.transform([comment]))

    # print(predicted_value)
    
    return predicted_value

def startpy():

    # predict_and_dump()
    # result = predict_comment('Good world!')
    # print('result : ', result)

    ml_obj = MLSingleton.getInstance()
    ml_obj.train_model(True)
    print(ml_obj.classify_single_comment('As long as you pay to keep them locked up for life, NOT THOSE OF US THAT WOULD NOT KEEP HIM ALIVE, you pay the taxes not me. I say hang them'))

    pass

if __name__ == "__main__":
    startpy()
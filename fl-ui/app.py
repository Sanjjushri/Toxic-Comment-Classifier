'''
Created on 

Course work: 

@author: Ana, Elakia, Sanjju

Source:
    
'''
from flask import Flask, render_template, jsonify, request
import json
import toxic_comment_classifier as toxic

app = Flask(__name__)

@app.route('/', methods = ['GET','POST'])
def start():


    final_list=[]

    Details = {}

    if request.method == 'POST':

        comment = request.form.get('comments')

        comments = json.loads(comment)

        out_file = open("data.json", "w") 
        
        json.dump(comments, out_file, indent = 6) 
        
        out_file.close() 

        my_file = open('data.json')
    
        comts = json.load(my_file)
        
        for element in comts['review_list']:

            cust_review = element['review']  
            result = toxic.predict_comment(cust_review)

            if result[0] ==1:
                final = "The comment is under review will update once verified."
                
            else:
                final = "The comment has been successfully updated"
                
            result = {
                       'review' : cust_review,
                       'result' : final
            }
            
            final_list.append(result)

            print(result)

            Details = {
                    'user_list' : final_list
                }

            print(final_list)


    return render_template('new.html', result = Details) 

    
        
if __name__ == "__main__":
    app.run(
        debug = True,
        host  = '0.0.0.0',
        port  = 3012
    )
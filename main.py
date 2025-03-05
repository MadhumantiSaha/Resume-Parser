from flask import Flask, render_template, request, redirect, url_for
import os
import PyPDF2
import docx2txt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



app = Flask(__name__)    #creating the flask app
app.config['UPLOAD_FOLDER'] = 'uploads/'  #creating the folder to store the uploaded files

#function to extract text from the pdf file
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)  #reading the pdf file
        for page in reader.pages:  #iterating over the pages
            text += page.extract_text()  #extracting the text from the page
        return text
    
def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
    
#
def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)        
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path) 
    else:
        return ""
        


@app.route("/")  #adding the html file to the main.py (frontend to backend)
def matchResume():
    return render_template('matchResume.html')  

@app.route("/matcher", methods=['GET', 'POST'])
def matcher():
    try:
        if request.method == 'POST':
            job_description = request.form.get('job_description')
            resume_files = request.files.getlist('resumes')

            if not resume_files or not job_description:
                return render_template('matchResume.html', message='Please upload resumes and enter a job description.')

            resumes = []
            for resume_file in resume_files:
                filename = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
                resume_file.save(filename)
                resumes.append(extract_text(filename))

            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform([job_description] + resumes)
            vectors_array = vectors.toarray()
            
            job_vector = vectors_array[0]
            resume_vectors = vectors_array[1:]
            
            similarity_scores = cosine_similarity([job_vector], resume_vectors)[0]
            
            # Get top 5 matches
            top_indices = similarity_scores.argsort()[-5:][::-1]
            top_resumes = [resume_files[i].filename for i in top_indices]
            top_scores = [similarity_scores[i] for i in top_indices]
            
            return render_template('matchResume.html', 
                                message='Matching complete!',
                                top_resumes=top_resumes,
                                similarity_scores=top_scores)
    except Exception as e:
        return render_template('matchResume.html', message=f'An error occurred: {str(e)}')
        
    return render_template('matchResume.html')

    #main part of the code
    # vectorizer = TfidfVectorizer()
    # vectors = vectorizer.fit_transform([job_description] + resumes)

    # job_vector = vectors[0]
    # resumes = vectors[1:]
    # print("=====================================")
    # print(resumes)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):  #checking if the folder exists
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
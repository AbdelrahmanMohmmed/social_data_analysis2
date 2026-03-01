import PyPDF2

file = open('PDF file.pdf','rb')
reader = PyPDF2.PdfReader(file)
page = reader.pages[0]
text = page.extract_text()
clean_text = " ".join(text.split())
print(clean_text)

clean_names = clean_text.split("Here are five student names:")[1].split("Here are five subjects:")[0]
clean_subjects = clean_text.split("Here are five subjects:")[1].split("Each student")[0]
print(clean_names)
print(clean_subjects)
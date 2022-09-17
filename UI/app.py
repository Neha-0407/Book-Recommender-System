import streamlit as st
from streamlit_lottie import st_lottie 
import pickle
import json
from predict import display_predict_page
from explore import display_explore_page

col1, col2 = st.columns(2)

with open("images/book.json", "r",errors='ignore') as f:
    book = json.load(f)


with open("images/books1.json", "r",errors='ignore') as f:
    books1 = json.load(f)


def header(url):
    st.markdown(f'<p style="text-align:center;color:white;font-weight:bolder;font-size:50px;">{url}</p>', unsafe_allow_html=True)

def subheader(url):
    st.markdown(f'<p style="text-align:center;color:#17252a;font-weight:bolder;font-size:25px;">{url}</p>', unsafe_allow_html=True)


with col1:
    header("Book Recommender System")

with col2:
    st_lottie(book)

with open('recommendbook.pkl', 'rb') as handle:
    knn = pickle.load(handle)
with open('dataset.pkl', 'rb') as handle:
    df = pickle.load(handle)

with open('books.pkl', 'rb') as handle:
    book_df = pickle.load(handle)

s = f"""
    <style>
    div.stButton > button:first-child {{ border: 3px solid #17252a;width: 200px; border-radius:10px 10px 10px 10px;  box-shadow: 0 10px 10px 0 rgba(0,0,0,0.50),0 10px 10px 0 rgba(0,0,0,0.50); }}
    div.stButton > button:first-child:hover {{ background-color:#3aafa9;width: 200px; box-shadow: 0 12px 16px 0 rgba(0,0,0,0.24),0 17px 50px 0 rgba(0,0,0,0.19);}}
    <style>
    """

book = st.text_input("Name Of The Book : ")
def searchBook(book):
    for i in range(2442):
        if(df.index[i]==book):
            return i
    return -1
btn = st.button("Search")

if btn:
    val = searchBook(book)
    col1, col2 = st.columns(2)
    if val!=-1:
        distances, indices = knn.kneighbors(df.iloc[val, :].values.reshape(1, -1), n_neighbors = 6)
        distances = distances.flatten()
        indices = indices.flatten()
        with col1: 
            for i in range(len(distances)):
                if i == 0:
                    st.write()
                else:
                    for j in range(1,len(distances)):
                        st.write(str(df.index[indices[j]]))
                    break
        with col2:
            st_lottie(books1)
    else:
        if val == -1:
            st.write("No such Book exist")
    


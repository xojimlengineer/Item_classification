import streamlit as st
from fastai.vision.all import*
import plotly.express as px
# import pathlib
# import platform
# plt=platform.system()
# if plt=='Linux':pathlib.WindowsPath=pathlib.PosixPath

# title
st.title("Transportni klassifikatsiya qiluvchi model")

# rasm yuklash
file=st.file_uploader("rasm yuklash",type=['png','jpeg','gif','svg','jfif'])
if file:
    st.image(file)

    #Pl convert
    img=PILImage.create(file)
    
    #model
    model=load_learner("transport_model.pkl")
    
    #prediction
    predic,index_id,pro=model.predict(img)
    st.success(f"Bashorat : {predic}")
    st.info(f"Ehtimollik:{pro[index_id]*100:.1f}%")

    # plotting
    fig=px.bar(x=pro*100,y=model.dls.vocab)
    st.plotly_chart(fig)

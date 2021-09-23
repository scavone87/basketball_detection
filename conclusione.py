import streamlit as st


def app():
    st.title("Conclusioni")

    st.markdown('''<div style="text-align: justify">
    Il progetto è stato realizzato per il corso di Visione e Percezione della facoltà di Ingegneria Informatica dell' Università dgli Studi della Basilicata.</div>
    <br>
    <div style="text-align: justify">
    In conclusione, l'approccio ideale è quello che utilizza le trasformate di Hough poiché, tranne che per la scelta dei vari parametri, non richiede alcun tipo di intervento manuale. 
    Questo però comporta dei risultati non sempre all'altezza delle aspettative. Di contro l'approccio manuale è in grado di fornire risultati soddisfacenti a discapito dell'automatizzazione del processo. 
    </div>''', unsafe_allow_html= True)

    st.subheader("Di seguito i link utili")
    st.markdown(''' 
    - [Google Colab](https://drive.google.com/drive/folders/1WRiGmXK0bP9Wff9xNXZKWLPSxFstdyZc?usp=sharing)
    - [GitHub](https://github.com/scavone87/basketball_detection)
    ''')
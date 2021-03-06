import streamlit as st
import pandas as pd
import sqlite3
from pickle import load

model=load(open('model.pkl','rb'))

cnx = sqlite3.connect('db.sqlite3')
# Get a cursor


# Execute a query def

def import_data():
    cur = cnx.cursor()
    cur.execute("SELECT Total_Trans_Ct, Total_Revolving_Bal, Total_Trans_Amt, Total_Relationship_Count, Total_Ct_Chng_Q4_Q1, Total_Amt_Chng_Q4_Q1, Customer_Age, Attrition_Flag FROM data_credit_card")
    data_card2 = cur.fetchall()
    cnx.close()
    return data_card2
data_card2 = import_data()
data_card_3 = pd.DataFrame(data_card2, columns=["Total_Trans_Ct", "Total_Revolving_Bal", "Total_Trans_Amt", "Total_Relationship_Count", "Total_Ct_Chng_Q4_Q1", "Total_Amt_Chng_Q4_Q1", "Customer_Age", "Attrition_Flag"])

st.title("Service de carte de crédit")

st.write("""
## Application de prédiction service carte de crédit
Cette application anticipe les susceptibles résiliation de la carte de crédit
""")
st.sidebar.header("Menu")
menu_name = st.sidebar.selectbox("", ("Accueil", "Importer données client", "Importer un DataSet"))

card = data_card_3[['Total_Trans_Ct',
                    'Total_Revolving_Bal',
                    'Total_Trans_Amt',
                    'Total_Relationship_Count',
                    'Total_Ct_Chng_Q4_Q1',
                    'Total_Amt_Chng_Q4_Q1',
                    'Customer_Age',
                    'Attrition_Flag']]

if menu_name == "Accueil":
    st.write('''## Bienvenue dans notre service
     Vous pouvez choisir une tâche dans le MENU''')
elif menu_name == "Importer données client":
    st.sidebar.header("Ajuster les informations sur le client")

    def user_input():
        Total_Trans_Ct = st.sidebar.slider('Nombre total de transactions (12 derniers mois)', 10, 139, 75)
        Total_Revolving_Bal = st.sidebar.slider('Solde renouvelable total', 0.0, 2517.0, 1250.0)
        Total_Trans_Amt = st.sidebar.slider('Montant total de la transaction (12 derniers mois)', 510.0, 18484.0,
                                            9200.0)
        Total_Relationship_Count = st.sidebar.slider('Nombre total de produits détenus', 1, 6, 3)
        Total_Ct_Chng_Q4_Q1 = st.sidebar.slider('Changement du nombre de transactions ', 0.0, 3.7140, 1.5)
        Total_Amt_Chng_Q4_Q1 = st.sidebar.slider('Changement du montant de la transaction', 0.0, 3.397, 1.6)
        Customer_Age = st.sidebar.slider('Âge du client en années', 26, 73, 35)

        data = {'Total_Trans_Ct': Total_Trans_Ct,
                'Total_Revolving_Bal': Total_Revolving_Bal,
                'Total_Trans_Amt': Total_Trans_Amt,
                'Total_Relationship_Count': Total_Relationship_Count,
                'Total_Ct_Chng_Q4_Q1': Total_Ct_Chng_Q4_Q1,
                'Total_Amt_Chng_Q4_Q1': Total_Amt_Chng_Q4_Q1,
                'Customer_Age': Customer_Age
                }

        client_info = pd.DataFrame(data, index=[1])

        return client_info

    df = user_input()

    st.subheader('On veut savoir si le client quitte le service de carte de crédit')
    st.write(df)

    y_pred = model.predict(df)

    st.subheader("Décision du client ")

    if y_pred == 1:
        st.write("!!! Attention !!! Le client est sur le point de départ")
    else:
        st.write("Un client satisfait n'a pas envie d'ailleurs")

else:
    st.write('')
    st.sidebar.header("Choisir le nombre de client ")
    nbre_client = st.sidebar.slider('Ajuster le nombre de client', 1, 200, 10)
    var_select = st.sidebar.slider('Varier la pioche (random_state)', 0, 1234, 0)

    card_mel = card.sample(n=nbre_client, random_state=var_select)
    X_cut = card_mel.drop('Attrition_Flag', axis=1)

    y_pred = model.predict(X_cut)

    def create_col(y_pr):
        y_pred_2 = []
        for elt in y_pr :
            if elt == 0:
                y_pred_2.append('Existing')
            else :
                y_pred_2.append('Attrited')

        return y_pred_2

    y_pred_3 = create_col(y_pred)

    st.subheader("Tableau complet des prédictions ")

    df_final_2 = card_mel.assign(Résultat = y_pred, prédiction = y_pred_3)
    st.write((df_final_2))

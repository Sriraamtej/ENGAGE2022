import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from streamlit_option_menu import option_menu


from joblib import load

from streamlit_lottie import st_lottie



df=pd.read_csv(r"C:\Users\srira\Downloads\cars_engage_2022.csv")

df['Car']= df['Make'].fillna('')+' '+df['Model']


def rmissingvaluecol(df, threshold):
    l = []
    l = list(df.drop(df.loc[:,list((100*(df.isnull().sum()/len(df.index)) >= threshold))].columns, 1).columns.values)
    print("# Columns having more than %s percent missing values: "%threshold, (df.shape[1] - len(l)))
    print("Columns:\n", list(set(list((df.columns.values))) - set(l)))
    return l


rmissingvaluecol(df,30) # Here threshold is 30% which means we are going to drop columns having more than 30% of missing values



dff=df[['Make','Model','Car','Variant','ARAI_Certified_Mileage','Body_Type','Fuel_Type','Fuel_System','Type','Drivetrain','Ex-Showroom_Price','Displacement','Cylinders',
     'Power','Torque','Fuel_Tank_Capacity','Height','Length','Width','Doors','Seating_Capacity','Wheelbase','Number_of_Airbags']]




dff['Ex-Showroom_Price']= dff['Ex-Showroom_Price'].str.replace(r'Rs.','')


dff['Ex-Showroom_Price'] = dff['Ex-Showroom_Price'].str.replace(r',', '')


dff["Ex-Showroom_Price"] = pd.to_numeric(dff["Ex-Showroom_Price"])

dff['Fuel_Tank_Capacity'] = dff['Fuel_Tank_Capacity'].str.replace(r'litres', '')


dff['Height'] = dff['Height'].str.replace(r'mm', '')
dff['Length'] = dff['Length'].str.replace(r'mm', '')
dff['Width'] = dff['Width'].str.replace(r'mm', '')
dff['Wheelbase'] = dff['Wheelbase'].str.replace(r'mm', '')
dff['Displacement'] = dff['Displacement'].str.replace(' cc','')
dff['ARAI_Certified_Mileage'] = dff['ARAI_Certified_Mileage'].str.replace(' km/litre','')

HP = dff.Power.str.extract(r'(\d{1,4}).*').astype(float) * 0.98632
HP = HP.apply(lambda x: round(x,2))
TQ = dff.Torque.str.extract(r'(\d{1,4}).*').astype(float)
TQ = TQ.apply(lambda x: round(x,2))
dff.Torque = TQ
dff.Power = HP

dff['Ex-Showroom_Price'] = dff['Ex-Showroom_Price'].astype(int)

dff['Height'] = dff['Height'].astype(float)
dff['Length'] = dff['Length'].astype(float)
dff['Width'] = dff['Width'].astype(float)
dff['Wheelbase'] = dff['Wheelbase'].astype(float)
dff['Displacement'] = dff['Displacement'].astype(float)

dff['ARAI_Certified_Mileage'] = dff['ARAI_Certified_Mileage'].replace('9.8-10.0','10')
dff['ARAI_Certified_Mileage']=dff['ARAI_Certified_Mileage'].replace('10kmpl','10')
dff['ARAI_Certified_Mileage'] = dff['ARAI_Certified_Mileage'].replace('22.4-21.9','22.4')

dff['ARAI_Certified_Mileage'] = dff['ARAI_Certified_Mileage'].astype(float)

dff = dff[~dff.ARAI_Certified_Mileage.isnull()]
dff = dff[~dff.Make.isnull()]
dff = dff[~dff.Width.isnull()]
dff = dff[~dff.Cylinders.isnull()]
dff = dff[~dff.Wheelbase.isnull()]

dff = dff[~dff['Seating_Capacity'].isnull()]
dff = dff[~dff['Torque'].isnull()]



dff.Number_of_Airbags.fillna(0,inplace= True)



dff['Fuel_Tank_Capacity']=dff['Fuel_Tank_Capacity'].astype(float)
dff['Fuel_Tank_Capacity']=dff['Fuel_Tank_Capacity'].fillna(dff['Fuel_Tank_Capacity'].mean())



fd=dff
from sklearn.cluster import KMeans
cols = [ i for i in fd.columns if fd[i].dtype != 'object']

# k-Means-Clustering
km = KMeans(n_clusters=8,)
clusters = km.fit_predict(fd[cols])
fd['cluster'] = clusters
fd.cluster = (fd.cluster + 1).astype('int64')




make_distribution=fd.groupby('Make').size()


def make_dist():
    fig=plt.figure(figsize=(15,8))
    make_distribution.plot(title= 'Make distribution')
    model=fd["Model"].value_counts().reset_index()
    model.columns=["model","count"]
    st.pyplot(fig)



def Model_plot():
    fig=plt.figure(figsize=(10,6))
    count_of_model = fd['Model'].value_counts()[0:10].sort_values(ascending=False)
    sns.countplot(data=fd, x='Model',order=count_of_model.index,palette="mako")
    plt.title('Frequency of model',size=15)
    plt.xlabel('Model',size=13)
    plt.ylabel('Count',size=13)
    st.pyplot(fig)
def Bodytype_plot():
    fig=plt.figure(figsize=(15,8))
    sns.boxplot(data=fd, x='Ex-Showroom_Price', y='Body_Type', palette='gist_rainbow')
    plt.title('Box plot of Ex-Showroom_Price Variation of every body type',fontsize=18)
    plt.ylabel('Body Type')
    st.pyplot(fig)  

def Fueltype_count():
    fig=plt.figure(figsize=(15,8))
    sns.countplot(data=fd, y='Fuel_Type',alpha=.6, color='r')
    plt.title('Cars count on differnt fuel type',fontsize=18)
    plt.xlabel('No of cars', fontsize=15)
    plt.ylabel('Fuel type', fontsize=15) 
    st.pyplot(fig)

def power_milage():
    fig=plt.figure(figsize=(15,8))
    ax=sns.scatterplot(data=fd, x='ARAI_Certified_Mileage',y='Power',color='g')
    plt.title('Power vs Milage',fontsize=15)
    ax.ticklabel_format(useOffset=False, style='plain')
    plt.xlabel('Milage',fontsize=15)
    st.pyplot(fig)

def milage_exshowrromprice():
    fig=plt.figure(figsize=(15,8))
    ax=sns.scatterplot(data=fd, x='ARAI_Certified_Mileage',y='Ex-Showroom_Price',color='Purple')
    plt.title('Ex_Showroom_price vs Milage',fontsize=15)
    ax.ticklabel_format(useOffset=False, style='plain')
    plt.xlabel('Milage',fontsize=15)
    st.pyplot(fig)
def power_price():
    fig=plt.figure(figsize=(15,8))
    ax=sns.scatterplot(data=fd, x='Power',y='Ex-Showroom_Price',hue='Body_Type',palette='gist_earth',alpha=.89, s=120 );
    plt.title('Horse power Vs Ex_Showroom_price',fontsize=15)
    ax.ticklabel_format(useOffset=False, style='plain')
    plt.ylabel('Ex_Showroom_price',fontsize=15)
    plt.xlabel('Horse power',fontsize=15)
    st.pyplot(fig)
def gridplot():
    fig=sns.pairplot(fd,vars=[ 'Displacement', 'ARAI_Certified_Mileage', 'Power', 'Ex-Showroom_Price'], hue= 'Fuel_Type',
             palette=sns.color_palette('magma',n_colors=4),diag_kind='kde',height=2, aspect=1.8)
    st.pyplot(fig)




decreg=load(r'C:\Users\srira\OneDrive\Desktop\ENGAGE\decreg.joblib')
def predict_note_authentication(ARAI_Certified_Mileage_input,Displacement_input,Power_input,Torque_input,Fuel_Tank_Capacity_input,Height_input,Length_input,Width_input,Doors_input,Seating_Capacity_input,Wheelbase_input,Number_of_Airbags_input,Body_Type_Sedan_input,Fuel_Type_Diesel_input,Fuel_Type_Hybrid_input):
    prediction=decreg.predict([[ARAI_Certified_Mileage_input,Displacement_input,Power_input,Torque_input,Fuel_Tank_Capacity_input,Height_input,Length_input,Width_input,Doors_input,Seating_Capacity_input,Wheelbase_input,Number_of_Airbags_input,Body_Type_Sedan_input,Fuel_Type_Diesel_input,Fuel_Type_Hybrid_input]])
    print(prediction)
    return prediction

def load_lottiefile(filepath:str):
    with open(filepath,"r") as f:
        return json.load(f)

lottie_insert=load_lottiefile(r"C:\Users\srira\OneDrive\Desktop\ENGAGE\hello.json")
lottie_insert2=load_lottiefile(r"C:\Users\srira\OneDrive\Desktop\ENGAGE\car-in-movement.json")
lottie_insert3=load_lottiefile(r"C:\Users\srira\OneDrive\Desktop\ENGAGE\tourists-by-car.json")
lottie_insert4=load_lottiefile(r"C:\Users\srira\OneDrive\Desktop\ENGAGE\data-processing.json")
lottie_insert5=load_lottiefile(r"C:\Users\srira\OneDrive\Desktop\ENGAGE\loading car-types.json")


def main():
    st.title("Car Data Analysis and Price Prediction")
    html_temp="""
    <div style="background-color:violet;padding:10px">
    <h2 style="color:white;text-align:center;">Car price predictor </h2>
    </div>
    """
    
          

    
    with st.sidebar:
        selected=option_menu(
        menu_title="Main Menu",
        options=["About","Feature analysis","Car Price Predictor"])
    if selected=="Feature analysis":
        st_lottie(
        lottie_insert2,
        speed=1,
        reverse=False,
        loop=True,
        quality="high",
        )
        st.header('An overview of key characteristics in a graphical format')
        st.write("Distribution of No of variants with respective to the model")
        if st.button("Model Plot"):
            Model_plot()
        st.write("Variation of Ex-Showroom_Price with respective to Body_Type")
        if st.button("BodyType Plot"):
            Bodytype_plot()
        st.write("Distribution of Manufacturer with respective to other")
        if st.button("Manufacturer Distribution"):
            make_distribution
            make_dist()
        st.write("No of cars based on Fuel_Type")
        if st.button("Count on fuel type"):
            Fueltype_count()
        st.write("Distribution of Power with respective to Milage")
        if st.button("power vs milage"):
            power_milage()
        st.write("Distribution of Milage With respective to Ex-Showroom_Price")
        if st.button("Milage Vs Ex-Showroom_Price"):
            milage_exshowrromprice()
        st.write("Distribution of Power With respective to Ex-Showroom_Price")
        if st.button("Power vs Ex-Showroom_Price"):
            power_price()
        st.write("Distribution of Milage,Displacement,Power and Ex-Showroom_Price")
        if st.button("Grid plot "):
            gridplot()
    if selected=="Car Price Predictor":
        st_lottie(
        lottie_insert3,
        speed=1,
        reverse=False,
        loop=True,
        quality="high",
        )
        st.markdown(html_temp,unsafe_allow_html=True)
        st.write("*Mandatory")
        ARAI_Certified_Mileage_input=st.text_input("ARAI_Certified_Mileage*","Type Here")
        Displacement_input=st.text_input("Displacement*","Type Here")
        Power_input=st.text_input("Power*","Type Here")
        Torque_input=st.text_input("Torque*","Type Here")
        Fuel_Tank_Capacity_input=st.text_input("Fuel_Tank_Capacity*","Type Here")
        Height_input=st.text_input("Height*","Type Here")
        Length_input=st.text_input("Length*","Type Here")
        Width_input=st.text_input("Width*","Type Here")
        Doors_input=st.text_input("Doors*","Type Here")
        Seating_Capacity_input=st.text_input("Seating_Capacity*","Type Here")
        Wheelbase_input=st.text_input("Wheelbase*","Type Here")
        Number_of_Airbags_input=st.text_input("Number_of_Airbags*","Type Here")
        Body_Type_Sedan_input=st.text_input("Body_Type_Sedan*","Type 1 if it is Sedan otherwise Type 0 ")
        Fuel_Type_Diesel_input=st.text_input("Fuel_Type_Diesel*","Type 1 if its fuel type is Diesel otherwise 0")
        Fuel_Type_Hybrid_input=st.text_input("Fuel_Type_Hybrid*","Type 1 if its fuel type is Hybrid otherwise 0")
        result=""
        if st.button("predict"):
            st_lottie(
            lottie_insert5,
            speed=1,
            reverse=False,
            loop=True,
            quality="high",
            )
            result=predict_note_authentication(ARAI_Certified_Mileage_input,Displacement_input,Power_input,Torque_input,Fuel_Tank_Capacity_input,Height_input,Length_input,Width_input,Doors_input,Seating_Capacity_input,Wheelbase_input,Number_of_Airbags_input,Body_Type_Sedan_input,Fuel_Type_Diesel_input,Fuel_Type_Hybrid_input)
            st.success('The predicted Ex-Showroom_Price Rs:{}'.format(result))
        
    if selected=="About":
        st_lottie(
        lottie_insert,
        speed=1,
        reverse=False,
        loop=True,
        quality="high",
        )
        st.header('What is this page??')
        st.write("This page contains some interesting analysis and predictive models that could assist the automobile industry in making better decisions about critical features and their values based on the price range in which they want to produce.This page was created with the Cars Data Set, which has over 1200 entries with over 120 features. ")
        st_lottie(
        lottie_insert4,
        speed=1,
        reverse=False,
        loop=True,
        quality="high",
        )
        st.write("On the following page, you can observe the relationships and variations among a variety of automotive features.")


if __name__=="__main__":
    main()


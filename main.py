import streamlit as st
from streamlit_option_menu import option_menu
import app


st.set_page_config(page_title="Story Builder")

class MultiApp:
    
    def __init__(self) -> None:
        self.apps=[]
    def add_app(self, title, function):
        self.apps.append({
            "title":title,
            "function":function
        })
    
    def run():
        apps= option_menu(
            menu_title=None,
            options=['Home','Functionality','User Story'],
            icons=['house-fill','gear','person'],
            default_index=0,
            orientation="horizontal",
            styles={
                "block-container":{"padding-top":"50px"},
                "container": {"margin":"0","padding": "0!important", "background-color": "#fafafa", "width":"100 %"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {
                    "font-size": "20px",
                    "text-align": "left",
                    "margin": "0px",
                    "color":"Black",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "green"},
            }
        )
        
        if(apps=="Home"):
            st.title("Welcome to Story Builder")
            app.main()

        elif(apps=="Functionality"):
            st.header("Fucntionality Gathered from the BRD :gear:")
            if "Components" not in app.st.session_state:
                st.warning("Upload your Business Requirement Document to see the functionality")
            else:
                st.write(app.st.session_state["Components"])
        
        elif(apps=="User Story"):
            st.header("User Story Generated based on the BRD")
            if "User Story" not in app.st.session_state:
                st.warning("Upload your Business Requirement Document to see the User Story")
            else:
                st.write(app.st.session_state["User Story"])
    
MultiApp.run()

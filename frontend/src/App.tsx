// App.tsx

import React, {useState, useEffect} from 'react';
import axios from 'axios';
import ClassificationList from './components/ClassificationList';
import TrainingMode from './components/TrainingMode';
import EmailManager from './components/EmailManager';

export interface Email {
    id: number;
    title: string;
    content: string;
    user_label: number | null; // Vom Nutzer gesetztes Label
    model_label: number | null; // Vom Modell vorhergesagtes Label
    prediction: number[] | null; // Wahrscheinlichkeit der Vorhersage
    trained: boolean; // Ob die E-Mail bereits trainiert wurde
}

export interface Classification {
    label: number;
    description: string;
}

const App = () => {
    const [emails, setEmails] = useState<Email[]>([]);
    const [classifications, setClassifications] = useState<Classification[]>([]);
    const [currentEmailIndex, setCurrentEmailIndex] = useState(0);
    const [staticTrainingEmails, setStaticTrainingEmails] = useState<Email[]>([]);

    const API_URL = 'http://127.0.0.1:8000';

    useEffect(() => {
        fetchEmails();
        fetchClassifications();
    }, []);

    const fetchEmails = async () => {
        try {
            const response = await axios.get(`${API_URL}/emails`);
            setEmails(response.data);
        } catch (error) {
            console.error('Fehler beim Abrufen der E-Mails:', error);
        }
    };

    const fetchClassifications = async () => {
        try {
            const response = await axios.get(`${API_URL}/classifications`);
            setClassifications(response.data);
        } catch (error) {
            console.error('Fehler beim Abrufen der Klassifikationen:', error);
        }
    };

    const startTraining = () => {
        const unlabelledEmails = emails.filter(
            (email) => email.user_label === null && email.prediction === null
        );

        if (unlabelledEmails.length === 0) {
            alert('Es gibt keine ungelabelten E-Mails für das Training.');
            return;
        }

        setStaticTrainingEmails(unlabelledEmails); // Fixiere die Liste
        setCurrentEmailIndex(0); // Zurücksetzen des Index
    };

    const moveToNextEmail = () => {
        setCurrentEmailIndex((prevIndex) => {
            const nextIndex = prevIndex + 1;

            if (nextIndex >= staticTrainingEmails.length) {
                setStaticTrainingEmails([]); // Liste leeren, wenn alle gelabelt sind
                return 0; // Zurücksetzen des Index
            }

            return nextIndex;
        });
    };

    return (
        <div className="container">
            <h1 className="text-center my-4">Email Ninja Frontend</h1>
            <button className="btn btn-info mb-3" onClick={startTraining}>
                Training starten
            </button>
            <div className="row">
                <div className="col-md-4">
                    <ClassificationList
                        classifications={classifications}
                        fetchClassifications={fetchClassifications}
                    />
                </div>
                <div className="col-md-4">
                    <TrainingMode
                        trainingEmails={staticTrainingEmails}
                        classifications={classifications}
                        currentEmailIndex={currentEmailIndex}
                        moveToNextEmail={moveToNextEmail}
                        totalEmails={staticTrainingEmails.length}
                        fetchEmails={fetchEmails}
                    />
                </div>
                <div className="col-md-4">
                    <EmailManager emails={emails} fetchEmails={fetchEmails}/>
                </div>
            </div>
        </div>
    );
};

export default App;
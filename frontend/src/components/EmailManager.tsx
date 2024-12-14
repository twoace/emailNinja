// EmailManager.tsx

import React from 'react';
import {Email} from '../App';
import axios from 'axios';

interface EmailManagerProps {
    emails: Email[];
    fetchEmails: () => void;
}

const EmailManager: React.FC<EmailManagerProps> = ({emails, fetchEmails}) => {
    const trainModel = async () => {
        try {
            await axios.post('http://127.0.0.1:8000/emails/train');
            alert('Modell erfolgreich trainiert!');
            fetchEmails();
        } catch (error) {
            console.error('Fehler beim Trainieren des Modells:', error);
        }
    };

    const classifyEmails = async () => {
        try {
            await axios.post('http://127.0.0.1:8000/classify');
            alert('E-Mails erfolgreich klassifiziert!');
            fetchEmails();
        } catch (error) {
            console.error('Fehler beim Klassifizieren der E-Mails:', error);
        }
    };

    const resetEmails = async () => {
        try {
            const response = await axios.post('http://127.0.0.1:8000/emails/reset');
            alert(response.data.message);
            fetchEmails(); // Aktualisiert die E-Mail-Liste
        } catch (error) {
            console.error('Fehler beim Zurücksetzen der E-Mails:', error);
        }
    };

    return (
        <div>
            <h3>Modellaktionen</h3>
            <button className="btn btn-success mb-2" onClick={trainModel}>
                Modell trainieren
            </button>
            <button className="btn btn-primary" onClick={classifyEmails}>
                E-Mails klassifizieren
            </button>
            <button className="btn btn-warning mb-2" onClick={resetEmails}>
                Labels zurücksetzen
            </button>
            <h4 className="mt-4">E-Mails Übersicht</h4>
            <ul className="list-group">
                {emails.map((email) => (
                    <li key={email.id} className={`list-group-item ${email.trained && email.user_label === email.model_label ? 'bg-success' : ''}`}>
                        <strong>{email.title}</strong><br/>
                        User Label: {email.user_label ?? 'N/A'} | Model Label: {email.model_label ?? 'N/A'} |
                        Prediction: {email.prediction ? email.prediction.join(", ") : 'N/A'}
                    </li>
                ))}
            </ul>
        </div>
    );
};

export default EmailManager;
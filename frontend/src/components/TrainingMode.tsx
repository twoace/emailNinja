// TrainingsMode.tsx

import React from 'react';
import {Classification, Email} from '../App';
import axios from 'axios';

interface TrainingModeProps {
    trainingEmails: Email[];
    classifications: Classification[];
    currentEmailIndex: number;
    moveToNextEmail: () => void;
    totalEmails: number;
    fetchEmails: () => void;
}

const TrainingMode: React.FC<TrainingModeProps> = ({
                                                       trainingEmails,
                                                       classifications,
                                                       currentEmailIndex,
                                                       moveToNextEmail,
                                                       totalEmails,
                                                       fetchEmails,
                                                   }) => {

    const countLabels = (): Record<number, number> => {
        const labelCounts: Record<number, number> = {};
        classifications.forEach((classification) => {
            labelCounts[classification.label] = 0;
        });

        trainingEmails.forEach((email) => {
            if (email.user_label !== null) {
                labelCounts[email.user_label] = (labelCounts[email.user_label] || 0) + 1;
            }
        });

        return labelCounts;
    };

    const labelCounts = countLabels();

    const labelEmail = async (emailId: number, label: number) => {
        try {
            await axios.post(`http://127.0.0.1:8000/emails/${emailId}/label`, {
                user_label: label,
            });

            // Lokale Aktualisierung der gelabelten E-Mail
            trainingEmails[currentEmailIndex].user_label = label;

            // Backend-Status aktualisieren, damit der EmailManager synchron bleibt
            fetchEmails();

            if (currentEmailIndex + 1 >= trainingEmails.length) {
                alert('Alle E-Mails wurden erfolgreich gelabelt.');
            }

            moveToNextEmail();
        } catch (error) {
            console.error('Fehler beim Labeln der E-Mail:', error);
        }
    };

    if (trainingEmails.length === 0) {
        return <p className="text-muted">Keine unlabelled E-Mails verfügbar.</p>;
    }

    if (currentEmailIndex >= trainingEmails.length) {
        return <p className="text-muted">Alle E-Mails wurden erfolgreich gelabelt.</p>;
    }

    const currentEmail = trainingEmails[currentEmailIndex];

    return (
        <div>
            <h3>Trainingsmodus</h3>
            <h5>
                E-Mail {currentEmailIndex + 1} von {totalEmails}
            </h5>
            <div className="mb-1">
                {classifications.map((classification) => (
                    <div key={classification.label}>
                        <button
                            className="btn btn-outline-primary me-2"
                            onClick={() => labelEmail(currentEmail.id, classification.label)}
                        >
                            {classification.description}
                        </button>
                        <span className="ms-2">
                            {labelCounts[classification.label] || 0} Zuordnungen
                        </span>
                    </div>
                ))}
            </div>
            <div className="card mb-3">
                <div className="card-header">
                    <strong>{currentEmail.title}</strong>
                </div>
                <div className="card-body">
                    <p>{currentEmail.content}</p>

                </div>
            </div>
        </div>
    );
};

export default TrainingMode;
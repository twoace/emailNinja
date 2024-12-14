import React, {useState, useEffect} from 'react';
import {Classification} from '../App';
import axios from 'axios';

interface ClassificationListProps {
    classifications: Classification[];
    fetchClassifications: () => void;
}

const ClassificationList: React.FC<ClassificationListProps> = ({classifications, fetchClassifications}) => {
    const [newClassification, setNewClassification] = useState<Classification>({label: 0, description: ''});

    useEffect(() => {
        // Automatically set the label to the next value
        if (classifications.length > 0) {
            const maxLabel = Math.max(...classifications.map((c) => c.label));
            setNewClassification((prev) => ({...prev, label: maxLabel + 1}));
        } else {
            setNewClassification((prev) => ({...prev, label: 0}));
        }
    }, [classifications]);

    const addClassification = async () => {
        try {
            await axios.post('http://127.0.0.1:8000/classifications', [newClassification]);
            fetchClassifications();
            setNewClassification((prev) => ({label: prev.label + 1, description: ''}));
        } catch (error) {
            console.error('Fehler beim Hinzufügen der Klassifikation:', error);
        }
    };

    return (
        <div>
            <h3>Klassifikationen</h3>
            <div className="mb-3">
                <input
                    type="number"
                    className="form-control mb-2"
                    placeholder="Label"
                    value={newClassification.label}
                    disabled
                />
                <input
                    type="text"
                    className="form-control mb-2"
                    placeholder="Beschreibung"
                    value={newClassification.description}
                    onChange={(e) => setNewClassification({...newClassification, description: e.target.value})}
                />
                <button className="btn btn-primary" onClick={addClassification}>
                    Klassifikation hinzufügen
                </button>
            </div>
            <ul className="list-group">
                {classifications.map((classification) => (
                    <li key={classification.label} className="list-group-item">
                        Label {classification.label}: {classification.description}
                    </li>
                ))}
            </ul>
        </div>
    );
};

export default ClassificationList;

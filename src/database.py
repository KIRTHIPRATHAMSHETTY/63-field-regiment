import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import os


class ScenarioDatabase:
    """SQLite database for storing scenario history."""
    
    def __init__(self, db_path: str = 'data/scenarios.db'):
        self.db_path = db_path
        self._ensure_db_exists()
    
    def _ensure_db_exists(self):
        """Create database and tables if they don't exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scenarios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                scenario_id TEXT NOT NULL UNIQUE,
                side_a_json TEXT NOT NULL,
                side_b_json TEXT NOT NULL,
                environment_json TEXT NOT NULL,
                predicted_outcome TEXT NOT NULL,
                prob_a REAL NOT NULL,
                prob_b REAL NOT NULL,
                prob_draw REAL NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_scenario(
        self,
        scenario_id: str,
        side_a: Dict[str, Any],
        side_b: Dict[str, Any],
        environment: Dict[str, Any],
        predicted_outcome: str,
        probabilities: Dict[str, float]
    ) -> int:
        """
        Save a scenario to the database.
        
        Args:
            scenario_id: Unique scenario identifier
            side_a: Side A parameters
            side_b: Side B parameters
            environment: Environment parameters
            predicted_outcome: Predicted outcome
            probabilities: Probability dictionary
        
        Returns:
            Row ID of inserted record
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT OR REPLACE INTO scenarios
            (timestamp, scenario_id, side_a_json, side_b_json, environment_json,
             predicted_outcome, prob_a, prob_b, prob_draw)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp,
            scenario_id,
            json.dumps(side_a),
            json.dumps(side_b),
            json.dumps(environment),
            predicted_outcome,
            probabilities.get('SIDE_A_ADVANTAGE', 0.0),
            probabilities.get('SIDE_B_ADVANTAGE', 0.0),
            probabilities.get('STALEMATE', 0.0)
        ))
        
        row_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return row_id
    
    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent scenario history.
        
        Args:
            limit: Maximum number of records to return
        
        Returns:
            List of scenario dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT scenario_id, timestamp, predicted_outcome, prob_a, prob_b, prob_draw
            FROM scenarios
            ORDER BY id DESC
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            scenario_id, timestamp, outcome, prob_a, prob_b, prob_draw = row
            
            # Parse timestamp for display
            try:
                dt = datetime.fromisoformat(timestamp)
                date_str = dt.strftime('%d %b %Y')
            except:
                date_str = timestamp
            
            # Determine blue win % (Side A is blue)
            blue_win_pct = int(prob_a * 100)
            
            # Format outcome
            if outcome == 'SIDE_A_ADVANTAGE':
                outcome_display = 'Blue Advantage'
            elif outcome == 'SIDE_B_ADVANTAGE':
                outcome_display = 'Red Advantage'
            else:
                outcome_display = 'Stalemate'
            
            history.append({
                'scenario_id': scenario_id,
                'date': date_str,
                'blue_win_pct': blue_win_pct,
                'outcome': outcome_display
            })
        
        return history
    
    def get_scenario(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific scenario by ID.
        
        Args:
            scenario_id: Scenario identifier
        
        Returns:
            Scenario dictionary or None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT scenario_id, timestamp, side_a_json, side_b_json, environment_json,
                   predicted_outcome, prob_a, prob_b, prob_draw
            FROM scenarios
            WHERE scenario_id = ?
        ''', (scenario_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        scenario_id, timestamp, side_a_json, side_b_json, environment_json, outcome, prob_a, prob_b, prob_draw = row
        
        return {
            'scenario_id': scenario_id,
            'timestamp': timestamp,
            'side_a': json.loads(side_a_json),
            'side_b': json.loads(side_b_json),
            'environment': json.loads(environment_json),
            'predicted_outcome': outcome,
            'probabilities': {
                'SIDE_A_ADVANTAGE': prob_a,
                'SIDE_B_ADVANTAGE': prob_b,
                'STALEMATE': prob_draw
            }
        }
    
    def clear_history(self):
        """Clear all scenario history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM scenarios')
        conn.commit()
        conn.close()


if __name__ == '__main__':
    # Test database
    print("Testing scenario database...")
    
    db = ScenarioDatabase()
    
    # Test save
    test_scenario = {
        'scenario_id': 'SCN-TEST-001',
        'side_a': {'force': 80, 'mobility': 0.75},
        'side_b': {'force': 70, 'mobility': 0.60},
        'environment': {'terrain': 'Hilly', 'weather': 'Cloudy'},
        'predicted_outcome': 'SIDE_A_ADVANTAGE',
        'probabilities': {'SIDE_A_ADVANTAGE': 0.72, 'SIDE_B_ADVANTAGE': 0.22, 'STALEMATE': 0.06}
    }
    
    row_id = db.save_scenario(
        test_scenario['scenario_id'],
        test_scenario['side_a'],
        test_scenario['side_b'],
        test_scenario['environment'],
        test_scenario['predicted_outcome'],
        test_scenario['probabilities']
    )
    
    print(f"Saved scenario with ID: {row_id}")
    
    # Test history
    history = db.get_history(limit=5)
    print(f"\nHistory ({len(history)} records):")
    for record in history:
        print(f"  {record['scenario_id']} | {record['date']} | {record['blue_win_pct']}% | {record['outcome']}")

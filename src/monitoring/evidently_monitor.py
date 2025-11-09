"""
Monitoring avec Evidently AI 
"""
import pandas as pd
import os
from datetime import datetime
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.test_suite import TestSuite
from evidently.legacy.tests import TestNumberOfDriftedColumns, TestShareOfDriftedColumns
import json

class EvidentlyMonitor:
    """Monitoring avec Evidently AI"""
    
    def __init__(self, reference_data_path):
        self.reference_data = pd.read_csv(reference_data_path)
        self.reports_dir = './logs/evidently_reports'
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def generate_drift_report(self, current_data_path):
        """Génère un rapport de drift avec Evidently"""
        current_data = pd.read_csv(current_data_path)
        
        # Créer le rapport
        report = Report(metrics=[
            DataDriftPreset()
        ])
        
        report.run(
            reference_data=self.reference_data,
            current_data=current_data
        )
        
        # Sauvegarder en HTML
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        html_path = f'{self.reports_dir}/drift_report_{timestamp}.html'
        report.save_html(html_path)
        
        print(f"✅ Rapport Evidently sauvegardé: {html_path}")
        
        # Extraire les métriques pour Prometheus
        report_dict = report.as_dict()
        return report_dict, html_path
    
    def run_drift_tests(self, current_data_path):
        """Exécute les tests de drift"""
        current_data = pd.read_csv(current_data_path)
        
        # Test suite - Utiliser des tests individuels au lieu de presets
        tests = TestSuite(tests=[
            TestNumberOfDriftedColumns(),
            TestShareOfDriftedColumns()
        ])
        
        tests.run(
            reference_data=self.reference_data,
            current_data=current_data
        )
        
        # Résultats
        results = tests.as_dict()
        
        # Sauvegarder
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_path = f'{self.reports_dir}/drift_tests_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Vérifier si tests passent
        all_passed = results.get('summary', {}).get('all_passed', False)
        
        if not all_passed:
            print("🚨 ALERTE: Drift détecté par Evidently!")
            self._generate_alert(results)
        else:
            print("✅ Pas de drift détecté")
        
        return results
    
    def _generate_alert(self, test_results):
        """Génère une alerte si drift détecté"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'source': 'evidently',
            'status': 'DRIFT_DETECTED',
            'all_tests_passed': False,
            'failed_tests': [
                test['name'] for test in test_results.get('tests', [])
                if test.get('status') == 'FAIL'
            ]
        }
        
        # Sauvegarder l'alerte
        alert_file = './logs/evidently_alerts.json'
        with open(alert_file, 'a') as f:
            f.write(json.dumps(alert) + '\n')
        
        print(f"⚠️ Alerte sauvegardée: {alert_file}")

def main():
    """Point d'entrée principal"""
    
    reference_path = "./data/processed/train_data.csv"
    current_path = "./data/processed/recent_data.csv"
    
    if not os.path.exists(reference_path):
        print("⚠️ Données de référence non trouvées")
        return
    
    if not os.path.exists(current_path):
        print("⚠️ Nouvelles données non trouvées")
        return
    
    print("🔍 Démarrage monitoring Evidently...")
    
    # Initialiser le monitor
    monitor = EvidentlyMonitor(reference_path)
    
    # Générer rapport de drift
    print("\n📊 Génération rapport de drift...")
    report_dict, html_path = monitor.generate_drift_report(current_path)
    
    # Exécuter tests de drift
    print("\n🧪 Exécution tests de drift...")
    test_results = monitor.run_drift_tests(current_path)
    
    print("\n✅ Monitoring Evidently terminé!")
    print(f"   - Rapport HTML: {html_path}")
    print(f"   - Ouvrir dans navigateur pour voir les détails")

if __name__ == "__main__":
    main()
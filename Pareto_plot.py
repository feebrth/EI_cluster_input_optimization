import optuna
from optuna.visualization import plot_pareto_front

# Lade die Studie
# study = optuna.load_study(study_name="GP_4", storage="mysql://optuna:password@127.0.0.1:3306/optuna_db")
#
# # Pareto-Front-Plot generieren
# fig = plot_pareto_front(study)
#
# # Schriftgrößen und Layout anpassen
# fig.update_layout(
#     font=dict(size=18),  # Allgemeine Schriftgröße
#     xaxis_title=dict(font=dict(size=18)),  # X-Achsenbeschriftung
#     yaxis_title=dict(font=dict(size=18)),  # Y-Achsenbeschriftung
#     legend=dict(font=dict(size=16))  # Schriftgröße der Legende
# )
#
# # Speichern der Plotly-Figur
# fig.write_image("pareto_4_GP.png", width=1200, height=800)  # Angepasste Bildgröße

import optuna

# Verbindung zur Datenbank herstellen
storage = "mysql://optuna:password@127.0.0.1:3306/optuna_db"

# Studien aus der Datenbank abrufen
try:
    summaries = optuna.get_all_study_summaries(storage=storage)
    if summaries:
        print("Studien in der Datenbank:")
        for summary in summaries:
            print(f"- Study ID: {summary.study_id}, Name: {summary.study_name}")
    else:
        print("Keine Studien in der Datenbank gefunden.")
except Exception as e:
    print(f"Fehler beim Abrufen der Studien: {e}")

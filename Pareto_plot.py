import optuna
from optuna.visualization import plot_pareto_front

# Lade die Studie
study = optuna.load_study(study_name="GP_24", storage="mysql://optuna:password@127.0.0.1:3306/optuna_db")

# Pareto-Front-Plot generieren
fig = plot_pareto_front(study)

# Schriftgrößen und Layout anpassen
fig.update_layout(
    font=dict(size=18),  # Allgemeine Schriftgröße
    xaxis_title=dict(font=dict(size=18)),  # X-Achsenbeschriftung
    yaxis_title=dict(font=dict(size=18)),  # Y-Achsenbeschriftung
    legend=dict(font=dict(size=16))  # Schriftgröße der Legende
)

# Speichern der Plotly-Figur
fig.write_image("pareto_4_RS.png", width=1200, height=800)  # Angepasste Bildgröße

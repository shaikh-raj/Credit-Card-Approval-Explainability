# Module for explainability (LIME and SHAP)

def explain_with_lime(explainer, model, X_test):
    """
    Function to explain the model with LIME
    """
    explanation = explainer.explain_instance(X_test, model.predict, num_features=5)
    explanation.show_in_notebook()

def explain_with_shap(explainer, model, X_test):
    """
    Function to explain the model with SHAP
    """
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test)

# LIME and SHAP examples can be added

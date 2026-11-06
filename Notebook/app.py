
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Draw
from rdkit.Chem import AllChem


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="ChromatoAI",
    page_icon="🧪",
    layout="wide"
)


# =====================================================
# 2. LOAD TRAINED MODEL
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "Model",
    "chromatography_model.joblib"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()

except FileNotFoundError:
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()


# =====================================================
# 3. MOLECULAR FEATURE EXTRACTION
# =====================================================

def calculate_features(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None, None

    features = {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "RotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "AromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol)
    }

    return mol, features


# =====================================================
# 4. MOLECULAR STRUCTURE GENERATION
# =====================================================

def draw_molecule(mol):

    mol = Chem.Mol(mol)

    AllChem.Compute2DCoords(mol)

    image = Draw.MolToImage(
        mol,
        size=(450, 350),
        kekulize=True
    )

    return image


# =====================================================
# 5. APPLICATION HEADER
# =====================================================

st.title("🧪 ChromatoAI")

st.subheader(
    "Machine Learning-Based Chromatography Solvent Selection"
)

st.markdown(
    """
    ChromatoAI uses molecular information and machine learning
    to predict suitable column chromatography solvent systems.

    Enter a molecule using its SMILES notation to begin.
    """
)

st.divider()


# =====================================================
# 6. MOLECULAR INPUT
# =====================================================

st.header("1. Molecular Input")

smiles = st.text_input(
    "Enter SMILES notation",
    placeholder="Example: CC(=O)c1ccccc1"
)

st.caption(
    "Enter a valid SMILES representation of your molecule."
)

predict_button = st.button(
    "🧪 Predict Solvent System",
    type="primary",
    use_container_width=True
)


# =====================================================
# 7. PREDICTION ENGINE
# =====================================================

if predict_button:

    if not smiles.strip():

        st.warning("Please enter a SMILES notation.")

    else:

        mol, features = calculate_features(smiles)

        if mol is None:

            st.error(
                "Invalid SMILES notation. Please check your input."
            )

        else:

            # ---------------------------------------------
            # PREPARE MODEL INPUT
            # ---------------------------------------------

            input_df = pd.DataFrame(
                [features],
                columns=[
                    "MolWt",
                    "LogP",
                    "TPSA",
                    "HBD",
                    "HBA",
                    "RotatableBonds",
                    "AromaticRings",
                    "FractionCSP3"
                ]
            )

            # ---------------------------------------------
            # CHECK MODEL FEATURE REQUIREMENTS
            # ---------------------------------------------

            expected_features = getattr(
                model,
                "feature_names_in_",
                None
            )

            if expected_features is not None:

                missing_features = [
                    col for col in expected_features
                    if col not in input_df.columns
                ]

                if missing_features:

                    st.error(
                        "Model feature mismatch. Missing features: "
                        + ", ".join(missing_features)
                    )
                    st.stop()

                input_df = input_df.reindex(
                    columns=expected_features
                )

            else:

                expected_count = getattr(
                    model,
                    "n_features_in_",
                    None
                )

                if expected_count is not None:
                    if expected_count != input_df.shape[1]:

                        st.error(
                            f"Model expects {expected_count} features, "
                            f"but the application generated "
                            f"{input_df.shape[1]} features."
                        )
                        st.stop()

            # ---------------------------------------------
            # GENERATE PREDICTION
            # ---------------------------------------------

            try:

                prediction = model.predict(input_df)[0]

            except Exception as e:

                st.error(f"Prediction failed: {e}")
                st.stop()

            # =================================================
            # 8. MOLECULAR ANALYSIS
            # =================================================

            st.divider()

            st.header("2. Molecular Analysis")

            col1, col2 = st.columns([1, 1])

            with col1:

                st.subheader("Molecular Structure")

                st.image(
                    draw_molecule(mol),
                    caption="2D Molecular Structure",
                    use_container_width=True
                )

                st.caption(f"SMILES: {smiles}")

            with col2:

                st.subheader("Molecular Descriptors")

                display_df = input_df.T.reset_index()

                display_df.columns = [
                    "Descriptor",
                    "Value"
                ]

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )

            # =================================================
            # 9. SOLVENT SYSTEM PREDICTION
            # =================================================

            st.divider()

            st.header("3. Predicted Solvent System")

            st.success(
                f"Predicted solvent system: {prediction}"
            )

            # =================================================
            # 10. PREDICTION PROBABILITIES
            # =================================================

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(input_df)[0]

                classes = model.classes_

                results = pd.DataFrame({
                    "Solvent System": classes,
                    "Probability (%)": probabilities * 100
                })

                results = results.sort_values(
                    by="Probability (%)",
                    ascending=False
                )

                st.subheader("Prediction Probability")

                st.bar_chart(
                    results.head(10).set_index("Solvent System")
                )

                st.dataframe(
                    results.head(10).round(2),
                    use_container_width=True,
                    hide_index=True
                )

            # =================================================
            # 11. SCIENTIFIC DISCLAIMER
            # =================================================

            st.info(
                """
                ChromatoAI provides machine-learning-based solvent
                system predictions. These predictions should be
                experimentally verified using TLC before column
                chromatography.
                """
            )


# =====================================================
# 12. SIDEBAR
# =====================================================

with st.sidebar:

    st.title("🧪 ChromatoAI")

    st.markdown("---")

    st.subheader("About")

    st.write(
        """
        An AI-assisted tool for column chromatography
        solvent system prediction using molecular descriptors.
        """
    )

    st.markdown("---")

    st.caption("Version 1.0")

    st.caption("Developed using Python, RDKit and Streamlit.")

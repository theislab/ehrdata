from __future__ import annotations

from typing import Any

import anndata as ad
import omop as op
from lamindb.core import AnnDataCurator
from lamindb_setup.core.types import UPathStr
from lnschema_core import Record
from lnschema_core.types import FieldAttr

from ehrdata import EHRData

from .lamin_config import DEFAULTS_VALUES

# TODO: Specify Lamin instance for validation in lamin_config.py


class EHRCurator(AnnDataCurator):
    """Custom curation flow for electronic health record data.

    This class handles the validation and preparation of electronic health record
    (EHR) data, ensuring that required columns are present and that data types
    are correct. It also validates concept IDs for consistency.

    Args:
        adata (ehrdata.EHRData | UPathStr): The AnnData object or path to the data
            containing the electronic health record information.
        categoricals (dict[str, FieldAttr], optional): A dictionary defining categorical
            fields and their attributes. Defaults to None.
        defaults (list[str], optional): A list of required columns that should be
            present in the data. Defaults to None.
        sources (dict[str, Record], optional): A dictionary specifying the sources
            of records. Defaults to None.
        organism (str, optional): The organism related to the dataset. Defaults to "human".
        concepts_var_column (str, optional): The column in `var` that contains measurement
            concept IDs to be validated. Defaults to "measurement_concept_id".
    """

    def __init__(
        self,
        edata: EHRData | UPathStr,
        categoricals: dict[str, FieldAttr] = None,
        *,
        defaults: dict[str, Any] = DEFAULTS_VALUES,
        sources: dict[str, Record] = None,
        organism: str = "human",
        concepts_var_column: str = "measurement_concept_id",
    ):
        """Initializes the EHRCurator with the provided parameters.

        This method sets up the EHRData object and relevant configurations for
        the curation process. It also calls the parent class's initializer.

        Args:
            edata (ehrdata.EHRData | UPathStr): The EHRData object or path containing the EHR data.
            categoricals (dict[str, FieldAttr], optional): Dictionary of categorical fields and their attributes.
            defaults (list[str], optional): List of required columns to validate in the `obs` data.
            sources (dict[str, Record], optional): Dictionary of sources for records.
            organism (str, optional): Organism related to the data. Defaults to "human".
            concepts_var_column (str, optional): Column in the `var` DataFrame containing measurement concept IDs.
        """
        self.edata = edata
        self.adata = ad.AnnData(X=edata.X, obs=edata.obs, var=edata.var)
        self.organism = organism
        self.defaults = defaults
        self.concepts_var_column = concepts_var_column

        super().__init__(data=self.adata, var_index=None, categoricals=categoricals, sources=sources, organism=organism)

    def _validate_obs(self):
        """Validate that required columns exist and have the correct types.

        This method checks the `obs` DataFrame to ensure that each column in `defaults`
        exists and that its values match the expected data type.

        Raises
        ------
            ValueError: If a required column is missing from the `obs` DataFrame.
            TypeError: If a column's data type does not match the expected type.
        """
        for column, expected_type in self.defaults.items():
            # Check if the column exists in the obs DataFrame
            if column not in self.adata.obs.columns:
                raise ValueError(f"Required column '{column}' is missing from the DataFrame.")

            # Check if the column's values have the expected data type using list comprehension
            if not all(isinstance(x, expected_type) for x in self.adata.obs[column]):
                raise TypeError(f"Column '{column}' has incorrect data type. Expected {expected_type.__name__}.")

    def validate_adata(self, op_instance) -> bool:
        """Run the validation process on the AnnData object.

        This method first validates the `obs` DataFrame, ensuring that all required
        columns and data types are correct. It then validates the measurement concept
        IDs in the `var` DataFrame, adding a new column `valid_concept_id` to indicate
        valid concept IDs.

        Returns
        -------
            bool: True if validation passes, False otherwise.
        """
        # First run custom validations
        self._validate_obs()

        # Validate concept IDs
        self.adata.var["valid_concept_id"] = op.Concept.validate(
            self.adata.var[self.concepts_var_column].values, field=op_instance.Concept.concept_id, mute=True
        )
        return EHRData(adata=self.adata)

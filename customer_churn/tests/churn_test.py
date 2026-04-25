# Insert churn model validation tests here. For example, you could test that the output of the validate_model function has the expected schema and value ranges.
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType


@pytest.fixture
def spark():
    """Create a Spark session for testing."""
    return SparkSession.builder.master("local[1]").appName("churn_test").getOrCreate()


@pytest.fixture
def valid_df(spark):
    """Create a valid DataFrame for testing."""
    schema = StructType([
        StructField("customer_id", StringType(), False),
        StructField("prediction", IntegerType(), False),
        StructField("probability", DoubleType(), False),
    ])
    data = [
        ("C001", 0, 0.1),
        ("C002", 1, 0.9),
        ("C003", 0, 0.3),
    ]
    return spark.createDataFrame(data, schema)


class TestValidateModel:
    """Tests for the validate_model function."""

    def test_valid_dataframe_passes(self, spark, valid_df):
        """Test that a valid DataFrame passes validation."""
        from churn_model.validate_model import validate_model
        result = validate_model(valid_df)
        assert result.count() == 3

    def test_empty_dataframe_raises_error(self, spark, valid_df):
        """Test that an empty DataFrame raises ValueError."""
        from churn_model.validate_model import validate_model
        empty_df = spark.createDataFrame([], valid_df.schema)
        
        with pytest.raises(ValueError, match="Input DataFrame is empty"):
            validate_model(empty_df)

    def test_missing_column_raises_error(self, spark, valid_df):
        """Test that missing required columns raise ValueError."""
        from churn_model.validate_model import validate_model
        schema = StructType([
            StructField("customer_id", StringType(), False),
            StructField("prediction", IntegerType(), False),
        ])
        df_missing_col = spark.createDataFrame([("C001", 0)], schema)
        
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_model(df_missing_col)

    def test_invalid_prediction_value_raises_error(self, spark):
        """Test that non-binary prediction values raise ValueError."""
        from churn_model.validate_model import validate_model
        schema = StructType([
            StructField("customer_id", StringType(), False),
            StructField("prediction", IntegerType(), False),
            StructField("probability", DoubleType(), False),
        ])
        data = [("C001", 2, 0.5)]  # prediction=2 is invalid
        df = spark.createDataFrame(data, schema)
        
        with pytest.raises(ValueError, match="invalid prediction values"):
            validate_model(df)

    def test_invalid_probability_range_raises_error(self, spark):
        """Test that probabilities outside [0,1] raise ValueError."""
        from churn_model.validate_model import validate_model
        schema = StructType([
            StructField("customer_id", StringType(), False),
            StructField("prediction", IntegerType(), False),
            StructField("probability", DoubleType(), False),
        ])
        data = [("C001", 0, 1.5)]  # probability=1.5 is invalid
        df = spark.createDataFrame(data, schema)
        
        with pytest.raises(ValueError, match="probability values outside"):
            validate_model(df)

    def test_null_values_raises_error(self, spark):
        """Test that null values in critical columns raise ValueError."""
        from churn_model.validate_model import validate_model
        schema = StructType([
            StructField("customer_id", StringType(), True),
            StructField("prediction", IntegerType(), True),
            StructField("probability", DoubleType(), True),
        ])
        data = [("C001", None, 0.5)]
        df = spark.createDataFrame(data, schema)
        
        with pytest.raises(ValueError, match="null values"):
            validate_model(df)

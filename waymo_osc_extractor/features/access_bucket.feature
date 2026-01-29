Feature: Access S3 Bucket

  Scenario: User accesses the S3 bucket successfully
    Given the user has valid AWS credentials
    When the user attempts to access the S3 bucket "waymo" with the prefix "tfrecords/training_tfexample.tfrecord"
    Then the user should see the list of objects in the bucket

  # Scenario: User fails to access the S3 bucket with invalid credentials
  #   Given the user has invalid AWS credentials
  #   When the user attempts to access the S3 bucket "waymo"
  #   Then the user should receive an access denied error
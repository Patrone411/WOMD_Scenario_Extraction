Feature: Access TFRecords in S3 Bucket

  Scenario: User accesses the scenarios in TFRecords successfully
    Given the user has valid AWS credentials
    When the user attempts to access the S3 bucket "waymo" with the prefix "tfrecords/training_tfexample.tfrecord"
    And the user attempts the list of tfrecords in the bucket
    And the user loads the tfrecord with the first key a the tmp file
    And the user attempts to enumerate scenarios
    Then the user should see that the scenario list is not empty
    And the first scenario should be taggable and stored with prefix in the env variable "RESULT_PREFIX"
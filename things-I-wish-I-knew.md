# Things I wish I knew about aws glue, spark and distributed systems

Originally posted: 08/20/20

<br>

From November 2019 to July 2020 I was working with a subscription based e-learning platform. With our subscription metrics on the rise, we needed new internal processes to keep up with the wealth of information we had on hand and were accumulating every day. This translated to reporting options that would allow management to have more autonomy in mining the data to support their initiatives so that our engineering team could focus on collecting cleaner data, and the principles, processes and techniques required to change the fundamental structure of our data driven initiatives. 

After putting together a statement of work, holding interviews with stakeholders, and transforming that qualitative information into quantitative information, my team designed the logical schema for my divisions data marts and data warehouse. Then we stood up a dc2.large Redshift cluster and the warehouses' physical schema, and we were ready to implement the data pipeline that would transfer our production data into an environment optimized for advanced analytics and interactive dashboards.

The design was simple...

<br/>

<img src = "https://user-images.githubusercontent.com/29679899/101289889-327fed00-37cd-11eb-8ee8-611a945f9f9d.png" width="700px">

<br/>

...but we ran into a few challenges that were pretty common that could trip up anyone attempting to take on such an ambitious task. It was my first time playing in the AWS sandbox, and there were many moments to learn.

<br/>

## Aws glue is not a one size fits all solution 

Glue is managed in Apache Spark, and it's not a fully mature ETL framework like Pentaho or Talend. There is a limit to the scalability of AWS Glue unless you're defining your logic in something more functional like pure Spark or Scala rather than Glue specific syntax, which is what we needed to do.

Glue also has issues its not entirely upfront about, such as their code structures needing to be organized in specific ways. There are soft limits on running concurrent jobs. It does not support reading/writing multiple dataframes in parallel from different data sources which means structuring the ETL for star schemas is not manageable in a meaningful way that allows for seamless workflow orchestration. In the end, we pivoted our design and just went with a flattened schema.

<br/>

## For file formatting, always use parquet over csv

To build scalable data pipelines, we need to switch from using local files, like CSVs, to distributed data sources, such as Parquet files on S3. We were loading tables from an OLTP database, which is a row store, to Redshift, which is a columnar database, so we needed the data to represent this optimization.

While the tools used across cloud platforms to load data vary significantly, the end result is usually the same, which is a dataframe. In a local environment, we can use Pandas to load the dataframe (csv), but in distributed environments we need different implementations such as Spark RDDs (parquet) in PySpark. Parquet is basically a columnar storage file format and it lets you read, compress and process only the columns required for the current query, whereas CSV files are row based.

<br/>

## Avoid expensive transformations and functions when validating your pipeline across large scale dataframes

When scaling our pipeline we ran into a few nontrivial issues. One of which required us to sift through Spark code and unit test their data partitioning functions. We needed to run a few jobs with dataframes at 75 to 100 million records a pop and when using native spark functions to handle our partitioning and the size of our clusters, we ran into speed efficiency issues. Each job was taking upwards of 24 to 48 hours to complete and some of them would just flat out stall.

We were partitioning on 120 to 144 RDDs at a 28 cluster per 6 node rate...

<br/>
 
# `numPartitions = numWorkerNodes * numCpu`
 
 <br/>

 ...which was mathematically sound, but our jobs were not behaving as expected. I started thinking there must be something we're not accounting for with the partitioning functions, given our run time had zero issues up until this point.

After investigating the source code, I figured out that one of the functions running against our production database was running a series of nested SQL functions, one of which included an MD5 checksum that was nesting a modulus generating various placeholder values to spread the data evenly across partitions based on the tables primary key, to which the MD5 function was calculating the key that we were partitioning on lots of times which slowed down the pipelines ability to run the job as fast as it could.

At that point the problem was as easy as removing the MD5 function so the modulus could just run on the key we were partitioning on and after that our jobs were able to run in a half hour with a lot less overhead.

<br/>

## Always tune your number of partitions

The number by which you partition your data will always be unique to your datasets. There is no one size fits all and for our use case we were only dealing with millions of comparisons so between 120 and 144 partitions got our larger jobs done. We also saw savings in cost by dynamically changing our workers and cores during job runs.

<br/>

## Partition on evenly distributed fields

A Spark application is executed in 3 steps:

1. An RDD graph is created, there will be more than one so this means we'll have a DAG (directed acyclic graph) of RDDs to represent the entire computation. 

2. The DAG scheduler divides operators into stages of tasks. 

3. A stage is composed on tasks based on partitions of the input data. The DAG scheduler pipelines operators together. Create stage graph, so a DAG of stages, that is a logical execution plan based on an RDD graph. Stages are created by breaking the RDD graph at shuffle boundaries.

Shuffle boundaries are important because they dictate how the data between workers are transported across a Spark clusters network. They basically redistribute data so it can be grouped differently across partitions. This operation is VERY expensive so its important that you partition your data based on an evenly distributed column like an id or key so the amount of data across your clusters are balanced. Be wary if you're working with funky UUID columns that overly represent particular values.

<br/>

## Visualizing the movement of your data will save you time

Not only will it save you time, but it will save you money and help you optimize your jobs. Diagnostic visualizations will give you insight into why jobs are failing, you can see how your data is distributed across your executors, how your data is moving, the shuffle of your partitions, and the cpu load between your driver and executors.

<br/>

## Conclusion

I honestly wish I had some of the insight here when I was asking myself, *"What am I doing wrong?"* or *"How can I make this run more efficiently"*. It reminds me of that saying, *"Experience is something you don't get until just after you need it"*. Regardless, the experience was great and we were able to deliver. While data engineering is sort of an under appreciated layer in the data science stack, I think businesses are realizing that without the correct plumbing they'll quickly find themselves in a classic garbage in garbage out scenario. 

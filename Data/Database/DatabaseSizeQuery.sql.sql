SELECT 
    table_schema AS DB_Name,
    ROUND(SUM(data_length + index_length) / 1024 / 1024, 2) AS "Database Size Megabytes(MB)"
FROM 
    information_schema.TABLES 
WHERE 
    table_schema = 'football' 
GROUP BY 
    table_schema;
SELECT 
	w.warehouse_id,
	w.warehouse_name,
	l.city,
	c.country_name,
	r.region_name
FROM warehouses w
INNER JOIN locations l
ON w.location_id = l.location_id
INNER JOIN countries c
ON l.country_id = c.country_id
INNER JOIN regions r
ON c.region_id = r.region_id
ORDER BY 1;



--

SELECT 
	p.product_name,
	i.quantity
FROM products p
INNER JOIN inventories i
ON p.product_id = i.product_id
INNER JOIN warehouses w
ON i.warehouse_id = w.warehouse_id
INNER JOIN locations l
ON w.location_id = l.location_id
INNER JOIN countries c
ON l.country_id = c.country_id
INNER JOIN regions r
ON c.region_id = r.region_id
WHERE r.region_name = 'Middle East and Africa'
ORDER BY quantity DESC;


SELECT 
	DISTINCT(w.warehouse_id)
	c.country_name
	r.region_name
FROM warehouses w
INNER JOIN locations l
ON w.location_id = l.location_id
INNER JOIN countries c
ON l.country_id = c.country_id
INNER JOIN regions r
ON c.region_id = r.region_id

------


SELECT 
	p.product_name,
	oi.quantity,
FROM products p
INNER JOIN order_items oi
ON p.product_id = oi.product_id
INNER JOIN orders o
ON oi.order_id = o.order_id
WHERE (EXTRACT(YEAR FROM o.order_date)) = 2016
ORDER BY 2 DESC;

SELECT 
	p.product_name,
	oi.quantity,
FROM products p
INNER JOIN order_items oi
ON p.product_id = oi.product_id
INNER JOIN orders o
ON oi.order_id = o.order_id
WHERE o.order_date BETWEEN '01/01/2016' AND '31/12/2016'
ORDER BY 2 DESC;



---------------------------------------------

SELECT 
	pc.category_name,
	oi.quantity
FROM product_categories pc
INNER JOIN products p
ON pc.category_id = p.category_id
INNER JOIN order_items oi
ON p.product_id = oi.product_id
INNER JOIN orders o
ON oi.order_id = o.order_id
WHERE o.order_date BETWEEN '01/01/2017' AND '31/12/2017'
ORDER BY 2 DESC;

SELECT 
	pc.category_name,
	oi.quantity
FROM product_categories pc
INNER JOIN products p
ON pc.category_id = p.category_id
INNER JOIN order_items oi
ON p.product_id = oi.product_id
INNER JOIN orders o
ON oi.order_id = o.order_id
WHERE (EXTRACT(YEAR FROM o.order_date)) = 2017
ORDER BY 2 DESC;



-----------------

SELECT 
	SUM(oi.quantity * oi.unit_price) AS sales,
	c.name
FROM order_items oi
INNER JOIN  orders o
ON oi.order_id = o.order_id
INNER JOIN customers c
ON o.customer_id = c.customer_id
WHERE (EXTRACT(YEAR FROM o.order_date)) = 2015
GROUP BY c.name
ORDER BY 1 DESC;

-----------------------

SELECT 
	SUM(oi.quantity * oi.unit_price) AS sales,
	EXTRACT(YEAR FROM o.order_date) 
FROM order_items oi
INNER JOIN orders o
ON oi.order_id = o.order_id
GROUP BY EXTRACT(YEAR FROM o.order_date) 
ORDER BY EXTRACT(YEAR FROM o.order_date) DESC;

SELECT 
	SUM(oi.quantity * oi.unit_price) AS sales,
	EXTRACT(YEAR FROM o.order_date) AS year 
FROM order_items oi
INNER JOIN orders o
ON oi.order_id = o.order_id
GROUP BY EXTRACT(YEAR FROM o.order_date)
ORDER BY 2 DESC;


-----------------------

SELECT 
	AVG(list_price) Average_price
FROM products;

SELECT 
	product_name
FROM products
WHERE list_price > 903,241
ORDER BY 1 DESC;

-----------------------------

SELECT 
	e.first_name,
	e.last_name, 
	SUM(oi.quantity * oi.unit_price) AS total_sales
FROM employees e 
INNER JOIN orders o
ON e.employee_id = o.salesman_id
INNER JOIN order_items oi
ON o.order_id = oi.order_id
WHERE EXTRACT(YEAR FROM o.order_date)= 2017 
GROUP BY e.first_name, e.last_name
HAVING SUM(oi.quantity * oi.unit_price) > 50000
ORDER BY 3 DESC;

------------------------------------

SELECT 
	COUNT(c.CUSTOMER_ID) AS Contact_number
FROM customers c
INNER JOIN contacts co
ON c.customer_id = co.customer_id


SELECT 
	c.name,
	co.first_name
FROM customers c
LEFT JOIN contacts co
ON c.customer_id = co.customer_id

----------------------------------------


SELECT 
	e.manager_id,
	SUM(io.quantity * io.unit_price) AS total_sales_manager
FROM employees e
INNER JOIN orders o
ON e.employee_id = o.salesman_id
INNER JOIN order_items io
ON o.order_id = io.order_id
WHERE EXTRACT(YEAR FROM o.order_date)= 2017
GROUP BY e.manager_id
ORDER BY 2 ASC;

SELECT 
	e.employee_id,
	COUNT(o.order_id) AS Number_of_sales
FROM employees e
LEFT JOIN orders o
ON e.employee_id = o.salesman_id
INNER JOIN order_items io
ON o.order_id = io.order_id
WHERE EXTRACT(YEAR FROM o.order_date) = 2017
GROUP BY e.employee_id
ORDER BY 1;

SELECT 
	employee_id,
	manager_id
FROM employees
WHERE employee_id IN (54,55,56,57,59,60,61,62,64)




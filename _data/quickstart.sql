SELECT 'CREATE DATABASE bdnb'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'bdnb')\gexec
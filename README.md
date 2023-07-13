# Deel-interview-assessment


Interview Assessment
This repository contains my submission for the interview coding assessment at [Company Name].

Overview
The goal of this assessment is to build a simple web application that allows users to:

View a list of movies
Search for movies by title
View details about a movie
Mark movies as a favorite
I chose to build this application using:

React for the frontend
Node/Express for the backend
MongoDB for the database
Running the Application
Backend
The backend is a simple Node/Express application. To run:

cd backend
npm install to install dependencies
npm start to start the server on http://localhost:5000
Frontend
The frontend is created with React and makes requests to the backend API. To run:

cd frontend
npm install to install dependencies
npm start to start the dev server on http://localhost:3000
Backend API
The backend exposes the following REST API endpoints:

GET /api/movies
Get a list of all movies.

GET /api/movies/:id
Get a single movie by id.

POST /api/movies
Create a new movie. Requires title, releaseYear, and rating in request body.

PUT /api/movies/:id
Update an existing movie.

DELETE /api/movies/:id
Delete a movie.

Frontend Features
The frontend allows users to:

View all movies on the home page
Search for movies by title on the home page
Click a movie to view details in a detail page
Favorite/unfavorite movies on the detail page
View all favorited movies on the favorites page
Testing
To run API tests:

npm test in the backend folder

To run frontend tests:

npm test in the frontend folder

Next Steps
If I had more time, here are some things I would work on next:

Adding user authentication
Allowing users to add ratings to movies
Building out a user profile page
Implementing pagination on the frontend
Adding more robust input validation
Improving test coverage
Please let me know if you have any other questions! I'm happy to clarify or expand on any part of this project.

# Case Study: Airbnb NYC Price Predictor

The problem. Pricing a short-term rental is one of the biggest pain
points for new Airbnb hosts. Price too high and a listing sits empty; price
too low and a host loses income every single night. Airbnb's own pricing
suggestions are often generic and don't account for the nuances of a
specific listing's location and booking profile — there's real value in a
tool that gives hosts a fast, data-driven price estimate before they even
publish their listing.

The approach. I built a machine learning pipeline using public NYC
Airbnb listings data, engineering features around location (borough), room
type, and booking activity (minimum nights, reviews, availability). After
comparing several regression models, a Gradient Boosting Regressor gave the
strongest, most reliable predictions. The final model is deployed as a live
Streamlit app, so a host can enter their listing's details and get an
instant suggested nightly price — no spreadsheet or data science background
required.

Business / real-world value. This kind of tool directly addresses a
revenue-optimization problem for a two-sided marketplace like Airbnb: hosts
earn more when pricing is accurate, guests get fairer prices, and the
platform benefits from higher occupancy and fewer abandoned listings. The
same underlying approach — predicting a fair market price from structured
listing attributes — generalizes well beyond Airbnb to other rental and
marketplace pricing problems (car rentals, event ticket resale, freelance
service pricing), making it a reusable pattern rather than a one-off
solution.

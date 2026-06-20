#!/usr/bin/env python3
"""Mock API server for the REST client task."""
from flask import Flask, jsonify

app = Flask(__name__)

USERS = [
    {"id": 1, "name": "Alice Johnson", "age": 34, "department": "Engineering", "salary": 95000.0},
    {"id": 2, "name": "Bob Smith", "age": 27, "department": "Marketing", "salary": 62000.0},
    {"id": 3, "name": "Carol White", "age": 41, "department": "Engineering", "salary": 110000.0},
    {"id": 4, "name": "Dave Brown", "age": 29, "department": "Sales", "salary": 58000.0},
    {"id": 5, "name": "Eve Davis", "age": 35, "department": "Marketing", "salary": 71000.0},
    {"id": 6, "name": "Frank Miller", "age": 52, "department": "Engineering", "salary": 130000.0},
    {"id": 7, "name": "Grace Wilson", "age": 23, "department": "Sales", "salary": 48000.0},
    {"id": 8, "name": "Henry Moore", "age": 38, "department": "HR", "salary": 75000.0},
    {"id": 9, "name": "Iris Taylor", "age": 31, "department": "HR", "salary": 68000.0},
    {"id": 10, "name": "Jack Anderson", "age": 45, "department": "Sales", "salary": 85000.0},
]

@app.route("/users")
def get_users():
    return jsonify(USERS)

@app.route("/users/<int:user_id>")
def get_user(user_id):
    user = next((u for u in USERS if u["id"] == user_id), None)
    if user is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(user)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

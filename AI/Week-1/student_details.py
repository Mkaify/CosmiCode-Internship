#!/usr/bin/env python3
"""
Student Details Management Program
This program collects student information and displays results using if-else and loops.
"""

def get_grade(percentage):
    """Determine grade based on percentage"""
    if percentage >= 90:
        return 'A+'
    elif percentage >= 80:
        return 'A'
    elif percentage >= 70:
        return 'B'
    elif percentage >= 60:
        return 'C'
    elif percentage >= 50:
        return 'D'
    else:
        return 'F'

def get_status(percentage):
    """Determine pass/fail status"""
    if percentage >= 50:
        return 'PASS'
    else:
        return 'FAIL'

def main():
    print("=" * 50)
    print("        STUDENT DETAILS MANAGEMENT SYSTEM")
    print("=" * 50)
    
    students = []
    
    while True:
        print("\n--- Enter Student Details ---")
        
        # Get student information
        name = input("Enter student name: ").strip()
        if not name:
            print("Name cannot be empty!")
            continue
            
        roll_no = input("Enter roll number: ").strip()
        if not roll_no:
            print("Roll number cannot be empty!")
            continue
        
        # Get marks for 5 subjects
        subjects = ['Math', 'Science', 'English', 'History', 'Computer Science']
        marks = []
        
        print(f"\nEnter marks for {name} (out of 100):")
        for subject in subjects:
            while True:
                try:
                    mark = float(input(f"{subject}: "))
                    if 0 <= mark <= 100:
                        marks.append(mark)
                        break
                    else:
                        print("Marks should be between 0 and 100!")
                except ValueError:
                    print("Please enter a valid number!")
        
        # Calculate total and percentage
        total_marks = sum(marks)
        percentage = total_marks / len(subjects)
        grade = get_grade(percentage)
        status = get_status(percentage)
        
        # Store student data
        student_data = {
            'name': name,
            'roll_no': roll_no,
            'subjects': subjects,
            'marks': marks,
            'total': total_marks,
            'percentage': percentage,
            'grade': grade,
            'status': status
        }
        students.append(student_data)
        
        # Ask if user wants to add more students
        choice = input("\nDo you want to add another student? (y/n): ").lower()
        if choice != 'y':
            break
    
    # Display all student results
    print("\n" + "=" * 80)
    print("                           STUDENT RESULTS")
    print("=" * 80)
    
    for i, student in enumerate(students, 1):
        print(f"\nStudent {i}:")
        print(f"Name: {student['name']}")
        print(f"Roll Number: {student['roll_no']}")
        print(f"Subject-wise Marks:")
        
        # Display marks for each subject
        for subject, mark in zip(student['subjects'], student['marks']):
            print(f"  {subject}: {mark}")
        
        print(f"Total Marks: {student['total']}/{len(student['subjects']) * 100}")
        print(f"Percentage: {student['percentage']:.2f}%")
        print(f"Grade: {student['grade']}")
        print(f"Status: {student['status']}")
        
        # Additional feedback using if-else
        if student['percentage'] >= 90:
            print("Excellent performance! Keep it up!")
        elif student['percentage'] >= 80:
            print("Very good performance!")
        elif student['percentage'] >= 70:
            print("Good performance!")
        elif student['percentage'] >= 60:
            print("Satisfactory performance. Room for improvement.")
        elif student['percentage'] >= 50:
            print("You passed, but need to work harder.")
        else:
            print("Failed. Please focus on studies and try again.")
        
        print("-" * 50)
    
    # Display class statistics
    if students:
        print(f"\n--- CLASS STATISTICS ---")
        total_students = len(students)
        passed_students = len([s for s in students if s['status'] == 'PASS'])
        failed_students = total_students - passed_students
        
        print(f"Total Students: {total_students}")
        print(f"Passed: {passed_students}")
        print(f"Failed: {failed_students}")
        print(f"Pass Percentage: {(passed_students/total_students)*100:.2f}%")
        
        # Find topper
        topper = max(students, key=lambda x: x['percentage'])
        print(f"Class Topper: {topper['name']} ({topper['percentage']:.2f}%)")

if __name__ == "__main__":
    main() 
"""
Todo Manager Interface
Interactive interface for managing AI workflow-based todos
"""

import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from ai_workflow_todos import AIWorkflowTodoManager, TodoStatus, TodoPriority, TodoItem

class TodoManagerInterface:
    """Interactive interface for managing todos"""
    
    def __init__(self):
        self.manager = AIWorkflowTodoManager()
        self.manager.create_all_workflows()
    
    def display_workflows(self):
        """Display all available workflows"""
        print("\n" + "="*60)
        print("AVAILABLE AI WORKFLOW TODOS")
        print("="*60)
        
        workflows = self.manager.get_all_workflows()
        for name, workflow in workflows.items():
            progress = self.manager.get_workflow_progress(name)
            print(f"\n{name.upper().replace('_', ' ')}")
            print(f"  Description: {workflow.description}")
            print(f"  Total Todos: {progress['total_todos']}")
            print(f"  Completed: {progress['completed_todos']}")
            print(f"  In Progress: {progress['in_progress_todos']}")
            print(f"  Pending: {progress['pending_todos']}")
            print(f"  Progress: {progress['completion_percentage']:.1f}%")
    
    def display_workflow_todos(self, workflow_name: str):
        """Display todos for a specific workflow"""
        workflow = self.manager.get_workflow(workflow_name)
        if not workflow:
            print(f"Workflow '{workflow_name}' not found")
            return
        
        print(f"\n{workflow.name.upper()}")
        print("="*60)
        print(f"Description: {workflow.description}")
        print(f"Category: {workflow.category}")
        print(f"Created: {workflow.created_at.strftime('%Y-%m-%d %H:%M')}")
        print(f"Status: {workflow.status.value}")
        
        print(f"\nTODOS ({len(workflow.todos)} total):")
        print("-" * 60)
        
        for i, todo in enumerate(workflow.todos, 1):
            status_icon = {
                TodoStatus.PENDING: "⏳",
                TodoStatus.IN_PROGRESS: "🔄",
                TodoStatus.COMPLETED: "✅",
                TodoStatus.CANCELLED: "❌",
                TodoStatus.BLOCKED: "🚫"
            }.get(todo.status, "❓")
            
            priority_icon = {
                TodoPriority.LOW: "🟢",
                TodoPriority.MEDIUM: "🟡",
                TodoPriority.HIGH: "🟠",
                TodoPriority.CRITICAL: "🔴"
            }.get(todo.priority, "⚪")
            
            print(f"\n{i:2d}. {status_icon} {priority_icon} {todo.title}")
            print(f"    Description: {todo.description}")
            print(f"    Category: {todo.category}")
            print(f"    Status: {todo.status.value}")
            print(f"    Priority: {todo.priority.value}")
            
            if todo.estimated_hours:
                print(f"    Estimated: {todo.estimated_hours}h")
            
            if todo.dependencies:
                print(f"    Dependencies: {', '.join(todo.dependencies)}")
            
            if todo.tags:
                print(f"    Tags: {', '.join(todo.tags)}")
            
            if todo.notes:
                print(f"    Notes: {len(todo.notes)} notes")
    
    def start_workflow(self, workflow_name: str):
        """Start a workflow by marking first todo as in progress"""
        workflow = self.manager.get_workflow(workflow_name)
        if not workflow:
            print(f"Workflow '{workflow_name}' not found")
            return
        
        # Find first pending todo
        first_todo = None
        for todo in workflow.todos:
            if todo.status == TodoStatus.PENDING:
                first_todo = todo
                break
        
        if not first_todo:
            print("No pending todos in this workflow")
            return
        
        # Mark as in progress
        first_todo.status = TodoStatus.IN_PROGRESS
        print(f"Started workflow '{workflow_name}'")
        print(f"Current todo: {first_todo.title}")
        print(f"Description: {first_todo.description}")
    
    def complete_todo(self, workflow_name: str, todo_id: str):
        """Mark a todo as completed"""
        workflow = self.manager.get_workflow(workflow_name)
        if not workflow:
            print(f"Workflow '{workflow_name}' not found")
            return
        
        todo = None
        for t in workflow.todos:
            if t.id == todo_id:
                todo = t
                break
        
        if not todo:
            print(f"Todo '{todo_id}' not found in workflow '{workflow_name}'")
            return
        
        todo.status = TodoStatus.COMPLETED
        todo.completed_at = datetime.now()
        
        print(f"Completed todo: {todo.title}")
        
        # Check if workflow is complete
        progress = self.manager.get_workflow_progress(workflow_name)
        if progress['completion_percentage'] == 100:
            print(f"🎉 Workflow '{workflow_name}' is now complete!")
        else:
            print(f"Progress: {progress['completion_percentage']:.1f}% complete")
    
    def add_todo_note(self, workflow_name: str, todo_id: str, note: str):
        """Add a note to a todo"""
        workflow = self.manager.get_workflow(workflow_name)
        if not workflow:
            print(f"Workflow '{workflow_name}' not found")
            return
        
        todo = None
        for t in workflow.todos:
            if t.id == todo_id:
                todo = t
                break
        
        if not todo:
            print(f"Todo '{todo_id}' not found in workflow '{workflow_name}'")
            return
        
        todo.notes.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M')}: {note}")
        print(f"Added note to '{todo.title}': {note}")
    
    def get_next_todos(self, workflow_name: str) -> List[TodoItem]:
        """Get next available todos (no dependencies or dependencies completed)"""
        workflow = self.manager.get_workflow(workflow_name)
        if not workflow:
            return []
        
        next_todos = []
        for todo in workflow.todos:
            if todo.status != TodoStatus.PENDING:
                continue
            
            # Check if all dependencies are completed
            dependencies_met = True
            for dep_id in todo.dependencies:
                dep_todo = None
                for t in workflow.todos:
                    if t.id == dep_id:
                        dep_todo = t
                        break
                
                if not dep_todo or dep_todo.status != TodoStatus.COMPLETED:
                    dependencies_met = False
                    break
            
            if dependencies_met:
                next_todos.append(todo)
        
        return next_todos
    
    def display_next_todos(self, workflow_name: str):
        """Display next available todos"""
        next_todos = self.get_next_todos(workflow_name)
        
        if not next_todos:
            print(f"No available todos in workflow '{workflow_name}'")
            return
        
        print(f"\nNEXT AVAILABLE TODOS in {workflow_name.upper()}:")
        print("-" * 50)
        
        for i, todo in enumerate(next_todos, 1):
            priority_icon = {
                TodoPriority.LOW: "🟢",
                TodoPriority.MEDIUM: "🟡",
                TodoPriority.HIGH: "🟠",
                TodoPriority.CRITICAL: "🔴"
            }.get(todo.priority, "⚪")
            
            print(f"\n{i}. {priority_icon} {todo.title}")
            print(f"   ID: {todo.id}")
            print(f"   Description: {todo.description}")
            print(f"   Category: {todo.category}")
            print(f"   Priority: {todo.priority.value}")
            
            if todo.estimated_hours:
                print(f"   Estimated: {todo.estimated_hours}h")
            
            if todo.tags:
                print(f"   Tags: {', '.join(todo.tags)}")
    
    def export_workflow(self, workflow_name: str, filename: str = None):
        """Export workflow to JSON file"""
        if not filename:
            filename = f"{workflow_name}_workflow.json"
        
        success = self.manager.export_workflow_to_json(workflow_name, filename)
        if success:
            print(f"Workflow '{workflow_name}' exported to '{filename}'")
        else:
            print(f"Failed to export workflow '{workflow_name}'")
    
    def interactive_menu(self):
        """Interactive menu for managing todos"""
        while True:
            print("\n" + "="*60)
            print("AI WORKFLOW TODO MANAGER")
            print("="*60)
            print("1. Display all workflows")
            print("2. Display workflow todos")
            print("3. Start workflow")
            print("4. Complete todo")
            print("5. Add todo note")
            print("6. Show next available todos")
            print("7. Export workflow")
            print("8. Exit")
            
            choice = input("\nEnter your choice (1-8): ").strip()
            
            if choice == "1":
                self.display_workflows()
            
            elif choice == "2":
                workflow_name = input("Enter workflow name: ").strip()
                self.display_workflow_todos(workflow_name)
            
            elif choice == "3":
                workflow_name = input("Enter workflow name: ").strip()
                self.start_workflow(workflow_name)
            
            elif choice == "4":
                workflow_name = input("Enter workflow name: ").strip()
                todo_id = input("Enter todo ID: ").strip()
                self.complete_todo(workflow_name, todo_id)
            
            elif choice == "5":
                workflow_name = input("Enter workflow name: ").strip()
                todo_id = input("Enter todo ID: ").strip()
                note = input("Enter note: ").strip()
                self.add_todo_note(workflow_name, todo_id, note)
            
            elif choice == "6":
                workflow_name = input("Enter workflow name: ").strip()
                self.display_next_todos(workflow_name)
            
            elif choice == "7":
                workflow_name = input("Enter workflow name: ").strip()
                filename = input("Enter filename (optional): ").strip()
                self.export_workflow(workflow_name, filename if filename else None)
            
            elif choice == "8":
                print("Goodbye!")
                break
            
            else:
                print("Invalid choice. Please try again.")

# Example usage
if __name__ == "__main__":
    interface = TodoManagerInterface()
    
    # Display all workflows
    interface.display_workflows()
    
    # Show research workflow todos
    interface.display_workflow_todos("research")
    
    # Show next available todos for research
    interface.display_next_todos("research")
    
    # Start interactive menu
    # interface.interactive_menu()

"""
title: Advanced Math Problem Solver Pipeline
author: Assistant
date: 2024-11-14
version: 2.0
license: MIT
description: A sophisticated math expression solver that provides detailed step-by-step solutions
requirements: pydantic
"""

from typing import List, Union, Generator, Iterator, Dict, Any
from pydantic import BaseModel
import ast
import operator
import logging
from decimal import Decimal, InvalidOperation
import re

class Pipeline:
    class Valves(BaseModel):
        """Configuration parameters for the math solver pipeline"""
        MAX_EXPRESSION_LENGTH: int = 200
        DECIMAL_PLACES: int = 4
        SHOW_STEPS: bool = True
        MAX_POWER: int = 10  # Prevent excessive computational load
        
    class MathNode:
        """Helper class for tracking mathematical operations"""
        def __init__(self, expression: str, value: float, operation: str = None):
            self.expression = expression
            self.value = value
            self.operation = operation
            
    def __init__(self):
        self.name = "Advanced Math Problem Solver"
        self.valves = self.Valves()
        self.operators = {
            ast.Add: (operator.add, '+'),
            ast.Sub: (operator.sub, '-'),
            ast.Mult: (operator.mul, '×'),
            ast.Div: (operator.truediv, '÷'),
            ast.Pow: (operator.pow, '^'),
        }
        
    async def on_startup(self):
        """Initialize the pipeline"""
        logging.info(f"Starting {self.name}")
        
    async def on_shutdown(self):
        """Cleanup pipeline resources"""
        logging.info(f"Shutting down {self.name}")

    def sanitize_expression(self, expression: str) -> str:
        """Clean and validate the mathematical expression"""
        # Remove whitespace and convert operators
        expression = re.sub(r'\s+', '', expression)
        expression = expression.replace('^', '**')
        
        # Basic validation
        if len(expression) > self.valves.MAX_EXPRESSION_LENGTH:
            raise ValueError(f"Expression too long (max {self.valves.MAX_EXPRESSION_LENGTH} characters)")
        
        # Check for invalid characters
        valid_chars = set('0123456789+-*/.()[]{}') 
        if not all(c in valid_chars for c in expression.replace('**', '')):
            raise ValueError("Expression contains invalid characters")
            
        return expression

    def format_number(self, number: float) -> str:
        """Format number to specified decimal places and remove trailing zeros"""
        try:
            decimal = Decimal(str(number))
            formatted = f"{decimal:.{self.valves.DECIMAL_PLACES}f}"
            # Remove trailing zeros after decimal point
            if '.' in formatted:
                formatted = formatted.rstrip('0').rstrip('.')
            return formatted
        except InvalidOperation:
            return str(number)

    def evaluate_node(self, node: ast.AST, steps: List[str]) -> MathNode:
        """Recursively evaluate AST nodes while tracking steps"""
        if isinstance(node, ast.Num):
            return self.MathNode(str(node.n), node.n)
            
        elif isinstance(node, ast.BinOp):
            # Get operator information
            op_func, op_symbol = self.operators[type(node.op)]
            
            # Evaluate left and right nodes
            left = self.evaluate_node(node.left, steps)
            right = self.evaluate_node(node.right, steps)
            
            # Perform operation
            try:
                if isinstance(node.op, ast.Div) and right.value == 0:
                    raise ValueError("Division by zero")
                    
                if isinstance(node.op, ast.Pow):
                    if right.value > self.valves.MAX_POWER:
                        raise ValueError(f"Power exceeds maximum allowed ({self.valves.MAX_POWER})")
                    
                result = op_func(left.value, right.value)
                
                # Format the expression
                expr = f"({left.expression} {op_symbol} {right.expression})"
                
                # Add step to solution
                if self.valves.SHOW_STEPS:
                    formatted_result = self.format_number(result)
                    steps.append(f"{expr} = {formatted_result}")
                
                return self.MathNode(expr, result, op_symbol)
                
            except (OverflowError, ValueError) as e:
                raise ValueError(f"Calculation error: {str(e)}")
                
        else:
            raise ValueError("Unsupported operation in expression")

    def solve_expression(self, expression: str) -> Dict[str, Any]:
        """Solve the mathematical expression and provide detailed solution"""
        try:
            # Sanitize and prepare expression
            clean_expr = self.sanitize_expression(expression)
            
            # Parse expression
            tree = ast.parse(clean_expr, mode='eval')
            
            # Track solution steps
            steps = []
            
            # Evaluate expression
            result = self.evaluate_node(tree.body, steps)
            
            # Format output
            formatted_result = self.format_number(result.value)
            
            return {
                'success': True,
                'original': expression,
                'result': formatted_result,
                'steps': steps if self.valves.SHOW_STEPS else None
            }
            
        except (SyntaxError, ValueError) as e:
            return {
                'success': False,
                'error': str(e),
                'original': expression
            }
        except Exception as e:
            logging.error(f"Unexpected error solving expression: {str(e)}")
            return {
                'success': False,
                'error': 'An unexpected error occurred',
                'original': expression
            }

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict
    ) -> Union[str, Generator, Iterator]:
        """Process the mathematical expression and return formatted solution"""
        
        # Handle title request
        if body.get("title", False):
            return self.name
            
        # Solve expression
        solution = self.solve_expression(user_message)
        
        # Format response
        if solution['success']:
            response = [
                f"Expression: {solution['original']}",
                f"Result: {solution['result']}"
            ]
            
            if solution['steps'] and len(solution['steps']) > 0:
                response.append("\nStep-by-step solution:")
                response.extend([f"{i+1}. {step}" for i, step in enumerate(solution['steps'])])
                
            return "\n".join(response)
        else:
            return f"Error: {solution['error']}"
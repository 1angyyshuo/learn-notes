---
title: JavaScript 闭包
date: 2024-04-11
created: 2024-04-11 20:00
tags: [javascript, 核心概念, 待复习]
category: 前端开发
status: 进行中
---

# JavaScript 闭包

## 概述

**闭包（Closure）** 是 JavaScript 中最重要但也最难理解的概念之一。它让函数可以"记住"并访问其词法作用域，即使函数在其词法作用域之外执行。

## 什么是闭包

### 简单定义

> 闭包是指有权访问另一个函数作用域中的变量的函数。

### 代码示例

```javascript
function outer() {
  let count = 0;  // 外部函数的变量

  // inner 函数就是一个闭包
  function inner() {
    count++;
    console.log(count);
  }

  return inner;
}

const counter = outer();  // outer 执行完毕，但 count 没有被销毁
counter();  // 1
counter();  // 2
counter();  // 3
```

## 闭包的核心原理

### 1. 作用域链

```javascript
function A() {
  let a = 'a';

  function B() {
    let b = 'b';

    function C() {
      let c = 'c';
      console.log(a, b, c);  // 可以访问所有外层变量
    }

    C();
  }

  B();
}
```

### 2. 变量生命周期

- 正常情况下，函数执行完毕后局部变量会被销毁
- **闭包让变量被引用而无法被垃圾回收**

```javascript
function createPerson(name) {
  // name 被内部函数引用，不会被销毁
  return {
    getName: function() {
      return name;
    },
    setName: function(newName) {
      name = newName;
    }
  };
}

const person = createPerson('张三');
console.log(person.getName());  // "张三"
person.setName('李四');
console.log(person.getName());  // "李四"
// name 变量依然存在！
```

## 实际应用场景

### 1. 数据私有化（模拟私有变量）

```javascript
const bankAccount = (function() {
  let balance = 0;  // 私有变量

  return {
    deposit: function(amount) {
      balance += amount;
      return balance;
    },
    withdraw: function(amount) {
      if (amount > balance) {
        throw new Error('余额不足');
      }
      balance -= amount;
      return balance;
    },
    getBalance: function() {
      return balance;
    }
  };
})();

bankAccount.deposit(100);
bankAccount.deposit(50);
console.log(bankAccount.getBalance());  // 150
// 无法直接访问 balance 变量
```

### 2. 函数工厂

```javascript
function createMultiplier(multiplier) {
  return function(number) {
    return number * multiplier;
  };
}

const double = createMultiplier(2);
const triple = createMultiplier(3);

console.log(double(5));  // 10
console.log(triple(5));  // 15
```

### 3. 防抖和节流

```javascript
// 防抖函数
function debounce(func, wait) {
  let timeout;

  return function(...args) {
    clearTimeout(timeout);
    timeout = setTimeout(() => {
      func.apply(this, args);
    }, wait);
  };
}

// 使用
const handleSearch = debounce((query) => {
  console.log('搜索:', query);
}, 500);

input.addEventListener('input', (e) => {
  handleSearch(e.target.value);
});
```

## 常见陷阱

### 循环中的闭包问题

```javascript
// ❌ 错误示例
for (var i = 1; i <= 5; i++) {
  setTimeout(function() {
    console.log(i);  // 全部都是 6
  }, i * 1000);
}

// ✅ 解决方案 1：使用 let
for (let i = 1; i <= 5; i++) {
  setTimeout(function() {
    console.log(i);  // 1, 2, 3, 4, 5
  }, i * 1000);
}

// ✅ 解决方案 2：使用 IIFE
for (var i = 1; i <= 5; i++) {
  (function(index) {
    setTimeout(function() {
      console.log(index);  // 1, 2, 3, 4, 5
    }, index * 1000);
  })(i);
}
```

### 内存泄漏

```javascript
// ❌ 可能导致内存泄漏
function leak() {
  const hugeData = new Array(1000000).fill('data');

  return function() {
    // 即使不使用 hugeData，它也不会被释放
    console.log('leak');
  };
}

// ✅ 正确使用
function noLeak() {
  return function(hugeData) {
    // 只在使用时传递数据
    console.log(hugeData.length);
  };
}
```

## 面试常见问题

### Q1: 闭包是什么？有什么优缺点？

**答：**
- **优点**：数据封装、实现模块化、保持状态
- **缺点**：内存占用、处理不当会导致内存泄漏

### Q2: 下面代码输出什么？

```javascript
function fn() {
  var arr = [];
  for (var i = 0; i < 3; i++) {
    arr[i] = function() {
      return i;
    };
  }
  return arr;
}

var result = fn();
console.log(result[0]());  // ?
console.log(result[1]());  // ?
console.log(result[2]());  // ?
```

**答案：都是 3**（因为共享同一个 `i`，且循环结束后 `i` 为 3）

## 深度理解

### 执行上下文与闭包的关系

```
全局执行上下文
  └─ outer 函数上下文
       └─ inner 函数上下文（闭包）
            └─ 引用 outer 的变量对象
```

### Chrome DevTools 查看闭包

1. 打开 DevTools → Sources
2. 设置断点在闭包函数内
3. Scope 面板可以看到 `Closure` 对象
4. 里面列出了闭包引用的所有变量

## 关联笔记

- [[javascript-this-绑定]] - this 的指向规则
- [[javascript-作用域链]] - 作用域与执行上下文
- [[javascript-原型链]] - 原型与继承机制
- [[前端性能优化]] - 内存管理与优化技巧

## 练习题目

1. 实现一个计数器函数，支持增减和重置
2. 使用闭包实现一个简单的缓存函数
3. 解释 React Hooks 中 useState 如何利用闭包

## 参考资料

- [MDN - 闭包](https://developer.mozilla.org/zh-CN/docs/Web/JavaScript/Closures)
- [JavaScript 高级程序设计（第4版）](https://book.douban.com/subject/35175321/)
- [You Don't Know JS](https://github.com/getify/You-Dont-Know-JS)

## 复习记录

- 2024-04-11: 首次学习，理解了基本概念和应用场景
- [ ] 2024-04-18: 一周后复习
- [ ] 2024-05-11: 一月后复习

---

💡 **核心要点**：闭包 = 函数 + 词法环境（引用的外部变量）

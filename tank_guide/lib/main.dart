import 'package:flutter/material.dart';
import 'auth_page.dart';

void main() {
  runApp(const TankGuideApp());
}

class TankGuideApp extends StatelessWidget {
  const TankGuideApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'AI Tank Guide',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: const AuthPage(),
    );
  }
}
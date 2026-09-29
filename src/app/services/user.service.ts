import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, Observable, tap } from 'rxjs';
import { User } from '../models/api.model';
import { api } from './api.util';

export const USER_STORAGE_KEY = 'mtgenesis.user';

/**
 * Username-only identity (spec §7). The logged-in user lives in localStorage so a
 * refresh keeps you signed in; every storage access is guarded because private
 * browsing and blocked storage can make localStorage throw.
 */
@Injectable({ providedIn: 'root' })
export class UserService {
  /** In-memory copy, used when localStorage is unavailable. */
  private memoryUser: User | null = null;
  private readonly userSubject: BehaviorSubject<User | null>;

  readonly user$: Observable<User | null>;

  constructor(private http: HttpClient) {
    this.memoryUser = this.readStorage();
    this.userSubject = new BehaviorSubject<User | null>(this.memoryUser);
    this.user$ = this.userSubject.asObservable();
  }

  login(username: string): Observable<User> {
    return this.http.post<User>(api('/users/login'), { username: username.trim() }).pipe(
      tap(user => this.store(user))
    );
  }

  currentUser(): User | null {
    const stored = this.readStorage();
    if (stored) {
      return stored;
    }
    return this.storageWorks() ? null : this.memoryUser;
  }

  logout(): void {
    this.memoryUser = null;
    try {
      localStorage.removeItem(USER_STORAGE_KEY);
    } catch {
      // storage unavailable: the in-memory copy is already cleared
    }
    this.userSubject.next(null);
  }

  private store(user: User): void {
    this.memoryUser = user;
    try {
      localStorage.setItem(USER_STORAGE_KEY, JSON.stringify(user));
    } catch {
      // storage unavailable: keep the in-memory copy for this session
    }
    this.userSubject.next(user);
  }

  private readStorage(): User | null {
    try {
      const raw = localStorage.getItem(USER_STORAGE_KEY);
      if (!raw) {
        return null;
      }
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed.id === 'string' && typeof parsed.username === 'string') {
        return { id: parsed.id, username: parsed.username };
      }
      return null;
    } catch {
      return null;
    }
  }

  private storageWorks(): boolean {
    try {
      localStorage.getItem(USER_STORAGE_KEY);
      return true;
    } catch {
      return false;
    }
  }
}
